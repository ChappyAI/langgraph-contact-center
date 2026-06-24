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
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

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
    def format(self, record):
        log_obj = {
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


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("LANG_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Service not configured: LANG_API_KEY not set")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


class InvokeRequest(BaseModel):
    action: str = Field(default="sentiment", description="sentiment | coaching | summary | qa | lead_score")
    transcript: str = Field(default="", max_length=MAX_TRANSCRIPT_CHARS)
    call_id: str = ""
    tenant_id: str = ""
    agent_id: str = ""
    duration_seconds: int = 0
    disposition: str = ""
    caller_number: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    ok: bool
    action: str
    call_id: str
    result: Dict[str, Any]
    error: Optional[str] = None


class RunsStreamRequest(BaseModel):
    """LangGraph platform-compatible payload (matches NestJS LangGraphConnector)."""

    assistant_id: str = Field(default="sentiment", description="Maps to State.action")
    input: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    stream_mode: str = "values"


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "lang-api", "version": "1.0.0"}


@app.get("/ok")
async def ok() -> Dict[str, str]:
    return {"status": "ok"}


def _build_state(action: str, payload: Dict[str, Any]):
    """Instantiate State dataclass from raw payload, defaulting safely."""
    from agent.graph import State

    transcript = str(payload.get("transcript", ""))[:MAX_TRANSCRIPT_CHARS]
    return State(
        action=action,
        transcript=transcript,
        call_id=str(payload.get("call_id", "")),
        tenant_id=str(payload.get("tenant_id", "")),
        agent_id=str(payload.get("agent_id", "")),
        duration_seconds=int(payload.get("duration_seconds", 0) or 0),
        disposition=str(payload.get("disposition", "")),
        caller_number=str(payload.get("caller_number", "")),
        metadata=payload.get("metadata") or {},
    )


def _serialize(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        return state
    if hasattr(state, "__dict__"):
        return {k: v for k, v in vars(state).items() if not k.startswith("_")}
    return {"value": state}


async def _run_graph(action: str, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    from agent.graph import graph
    from api.tracing import current_call_id
    
    # Set the ContextVar so any deep LLM calls can pull the call_id
    call_id = str(payload.get("call_id", "unknown_call"))
    token = current_call_id.set(call_id)

    state = _build_state(action, payload)
    
    # Inject call_id and action into LangSmith metadata for filtering
    run_config = {
        "configurable": context,
        "metadata": {
            "call_id": call_id, 
            "action": action,
            "tenant_id": str(payload.get("tenant_id", ""))
        }
    }
    
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(state, config=run_config),
            timeout=INVOKE_TIMEOUT_SECONDS,
        )
    finally:
        current_call_id.reset(token)
        
    return _serialize(result)


@app.post("/invoke", response_model=InvokeResponse, dependencies=[Depends(require_api_key)])
async def invoke(req: InvokeRequest) -> InvokeResponse:
    """Direct invoke — simple JSON request/response."""
    payload = req.model_dump()
    context = {
        "tenant_id": req.tenant_id,
        "call_id": req.call_id,
        "agent_id": req.agent_id,
        "action": req.action,
    }
    try:
        result = await _run_graph(req.action, payload, context)
    except asyncio.TimeoutError:
        logger.warning("graph timeout (call_id=%s)", req.call_id)
        raise HTTPException(status_code=504, detail="graph invocation timed out")
    except Exception as exc:
        logger.exception("Graph invocation failed (call_id=%s)", req.call_id)
        return InvokeResponse(
            ok=False,
            action=req.action,
            call_id=req.call_id,
            result={},
            error=str(exc),
        )

    err = result.get("error") if isinstance(result, dict) else None
    return InvokeResponse(
        ok=err is None,
        action=req.action,
        call_id=req.call_id,
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
