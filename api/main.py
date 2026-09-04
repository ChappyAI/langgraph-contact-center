"""FastAPI wrapper exposing the LangGraph agent over HTTP for Fly.io.

Endpoints:
- GET  /health      — liveness
- GET  /ok          — readiness (used by backend health check)
- POST /invoke      — direct invoke (custom payload)
- POST /runs/stream — LangGraph-platform-compatible SSE endpoint for legacy clients

Auth: x-api-key header validated against LANG_API_KEY env (skipped if unset for dev/internal flycast).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import logfire
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="chappy-lang-api",
    send_to_logfire="if-token-present",
)


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for stdout aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a single-line JSON object."""
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "langgraph-contact-center",
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO"))

logger = logging.getLogger(__name__)

INVOKE_TIMEOUT_SECONDS = float(os.getenv("INVOKE_TIMEOUT_SECONDS", "55"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "50000"))
TRANSCRIPT_ANALYSIS_ACTIONS = {"sentiment", "coaching", "summary", "lead_score"}
_RESULT_CACHE: Dict[tuple[str, str], tuple[Dict[str, Any], Any]] = {}
_RESULT_CACHE_LOCK = asyncio.Lock()


def _check_api_key_configured() -> None:
    """Validate LANG_API_KEY is set at import time / startup."""
    if not os.getenv("LANG_API_KEY"):
        raise RuntimeError(
            "LANG_API_KEY environment variable is required but not set. "
            "Refusing to start without authentication configured."
        )


_check_api_key_configured()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: validate env on startup, log shutdown."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — LLM nodes will fail.")
    logger.info("lang-api started on port %s", os.getenv("PORT", "8002"))
    yield
    logger.info("lang-api shutting down")


app = FastAPI(
    title="Chappy Connect LangGraph API",
    version="1.0.0",
    lifespan=lifespan,
)
logfire.instrument_fastapi(app)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce x-api-key header against LANG_API_KEY for protected routes."""
    expected = os.getenv("LANG_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Service not configured: LANG_API_KEY not set")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


class InvokeRequest(BaseModel):
    """Payload for the /invoke endpoint — one call's worth of state."""

    action: str = Field(default="sentiment", description="sentiment | coaching | summary | qa | lead_score")
    transcript: str = Field(default="", max_length=MAX_TRANSCRIPT_CHARS)
    call_id: str | None = None
    tenant_id: str | None = None
    agent_id: str | None = None
    duration_seconds: int | None = None
    disposition: str | None = None
    hangup_cause: str | None = None
    lead_id: str | None = None
    campaign_id: str | None = None
    campaign_outcome: str | None = None
    caller_number: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    supersede: bool = False
    version: str | int | None = None


class InvokeResponse(BaseModel):
    """Response envelope for the /invoke endpoint."""

    ok: bool
    action: str
    call_id: str
    result: Dict[str, Any]
    error: str | None = None


class RunsStreamRequest(BaseModel):
    """LangGraph platform-compatible payload (matches NestJS LangGraphConnector)."""

    assistant_id: str = Field(default="sentiment", description="Maps to State.action")
    input: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    stream_mode: str = "values"


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness probe returning service metadata."""
    return {"status": "ok", "service": "lang-api", "version": "1.0.0"}


@app.get("/ok")
async def ok() -> Dict[str, str]:
    """Lightweight readiness probe used by the backend health check."""
    return {"status": "ok"}


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _assemble_call_context(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the canonical call context."""
    normalized = dict(payload)
    disposition = normalized.get("disposition")
    hangup_cause = normalized.get("hangup_cause")
    if _missing(disposition) and not _missing(hangup_cause):
        normalized["disposition"] = hangup_cause

    if action in TRANSCRIPT_ANALYSIS_ACTIONS:
        missing = [
            name
            for name in ("call_id", "tenant_id", "duration_seconds", "lead_id")
            if _missing(normalized.get(name))
        ]
        if _missing(normalized.get("disposition")):
            missing.append("disposition or hangup_cause")
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"incomplete call context: missing {', '.join(missing)}",
            )
    return normalized


def _build_state(action: str, payload: Dict[str, Any]):
    """Instantiate State from the canonical call context."""
    from agent.graph import State

    payload = _assemble_call_context(action, payload)
    transcript = str(payload.get("transcript", ""))[:MAX_TRANSCRIPT_CHARS]
    return State(
        action=action,
        transcript=transcript,
        call_id=payload.get("call_id") or "",
        tenant_id=payload.get("tenant_id") or "",
        agent_id=payload.get("agent_id") or "",
        duration_seconds=payload.get("duration_seconds") or 0,
        disposition=payload.get("disposition") or "",
        lead_id=payload.get("lead_id") or "",
        campaign_id=payload.get("campaign_id") or "",
        campaign_outcome=payload.get("campaign_outcome") or "",
        caller_number=payload.get("caller_number") or "",
        metadata=payload.get("metadata") or {},
    )


def _serialize(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        return state
    if hasattr(state, "__dict__"):
        return {k: v for k, v in vars(state).items() if not k.startswith("_")}
    return {"value": state}


async def _run_graph(
    action: str, payload: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    from agent.graph import graph
    from api.tracing import current_call_id

    normalized = _assemble_call_context(action, payload)
    call_id = str(normalized.get("call_id") or "unknown_call")
    cache_key = (call_id, action)
    supersede = (
        bool(normalized.get("supersede")) or normalized.get("version") is not None
    )

    async with _RESULT_CACHE_LOCK:
        cached = _RESULT_CACHE.get(cache_key)
        if cached is not None and not supersede:
            return copy.deepcopy(cached[0])

        token = current_call_id.set(call_id)
        state = _build_state(action, normalized)
        run_config = {
            "configurable": context,
            "metadata": {
                "call_id": call_id,
                "action": action,
                "tenant_id": normalized.get("tenant_id") or "",
            },
        }
        try:
            result = await asyncio.wait_for(
                graph.ainvoke(state, config=run_config),
                timeout=INVOKE_TIMEOUT_SECONDS,
            )
        finally:
            current_call_id.reset(token)

        serialized = _serialize(result)
        if cached is not None:
            serialized = {**serialized, "superseded_previous_result": True}
        if not serialized.get("error"):
            _RESULT_CACHE[cache_key] = (
                copy.deepcopy(serialized),
                normalized.get("version"),
            )
        return serialized


@app.post("/invoke", response_model=InvokeResponse, dependencies=[Depends(require_api_key)])
async def invoke(req: InvokeRequest) -> InvokeResponse:
    """Direct invoke — simple JSON request/response."""
    payload = req.model_dump()
    payload = _assemble_call_context(req.action, payload)
    context = {
        "tenant_id": req.tenant_id or "",
        "call_id": req.call_id or "",
        "agent_id": req.agent_id,
        "action": req.action,
    }
    try:
        result = await _run_graph(req.action, payload, context)
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.warning("graph timeout (call_id=%s)", req.call_id)
        raise HTTPException(status_code=504, detail="graph invocation timed out")
    except Exception as exc:
        logger.exception("Graph invocation failed (call_id=%s)", req.call_id)
        return InvokeResponse(
            ok=False,
            action=req.action,
            call_id=req.call_id or "",
            result={},
            error=str(exc),
        )

    err = result.get("error") if isinstance(result, dict) else None
    return InvokeResponse(
        ok=err is None,
        action=req.action,
        call_id=req.call_id or "",
        result=result,
        error=err,
    )


@app.post("/runs/stream", dependencies=[Depends(require_api_key)])
async def runs_stream(req: RunsStreamRequest):
    """LangGraph platform-compatible SSE endpoint.

    Streams final state as a single `data: {json}` chunk followed by `[DONE]`.
    The NestJS connector accumulates chunks; only the last one is used.
    """
    action = req.assistant_id or "sentiment"
    payload = dict(req.input or {})
    payload = _assemble_call_context(action, payload)
    context = (req.config or {}).get("configurable", {}) if isinstance(req.config, dict) else {}
    context = {**context, "action": action}

    async def event_stream():
        try:
            result = await _run_graph(action, payload, context)
            yield f"event: values\ndata: {json.dumps(result)}\n\n"
            yield "data: [DONE]\n\n"
        except asyncio.TimeoutError:
            logger.warning("graph timeout (call_id=%s)", payload.get("call_id"))
            yield f"event: error\ndata: {json.dumps({'error': 'graph invocation timed out'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Graph stream failed (call_id=%s)", payload.get("call_id"))
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
