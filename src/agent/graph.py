"""Con.Nexus Contact Center AI Agent — LangGraph implementation.

Multi-step agent for contact center operations:
- Analyze call transcripts for sentiment and coaching (LLM-powered)
- Score leads from call data
- Generate post-call summaries (LLM-powered)
- Evaluate call quality
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.models import get_llm_chain, is_terminal_provider_error, resolve_llm_config


class AllProvidersFailed(Exception):
    """Raised when every configured LLM provider fails."""

    def __init__(self, last_error: BaseException, attempted_providers: list[str]):
        """Store the last provider error and attempted providers."""
        self.last_error = last_error
        self.attempted_providers = attempted_providers
        super().__init__(
            f"All LLM providers failed ({', '.join(attempted_providers)}): {last_error}"
        )


class Context(TypedDict, total=False):
    """Runtime context for the contact center agent."""

    tenant_id: str
    call_id: str
    agent_id: str
    action: str
    provider: str
    model: str
    fallback_provider: str
    fallback_model: str


@dataclass
class State:
    """State flowing through the contact center agent graph."""

    action: str = "sentiment"
    transcript: str = ""
    call_id: str = ""
    tenant_id: str = ""
    agent_id: str = ""
    duration_seconds: int = 0
    disposition: str = ""
    caller_number: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    fallback_provider: str = ""
    fallback_model: str = ""

    sentiment: Dict[str, Any] | None = None
    coaching_tip: str | None = None
    summary: Dict[str, Any] | None = None
    qa_score: Dict[str, Any] | None = None
    lead_score: Dict[str, Any] | None = None
    routing: Dict[str, Any] | None = None
    error: str | None = None


async def _invoke_with_fallback(
    prompt: str, state: State, context: Context | None = None
) -> Any:
    """Invoke configured providers with terminal classification and retries."""
    request_context = context or {}
    provider, model, fallback_provider, fallback_model = resolve_llm_config(
        state.provider or request_context.get("provider"),
        state.model or request_context.get("model"),
        state.fallback_provider or request_context.get("fallback_provider"),
        state.fallback_model or request_context.get("fallback_model"),
    )
    providers = [provider]
    if (fallback_provider, fallback_model) != (provider, model):
        providers.append(fallback_provider)

    last_error: BaseException | None = None
    for index, llm in enumerate(
        get_llm_chain(
            provider,
            model,
            temperature=0.3,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )
    ):
        for attempt in range(3):
            try:
                return await llm.ainvoke(prompt)
            except Exception as exc:
                last_error = exc
                if is_terminal_provider_error(exc) or attempt == 2:
                    break
                await asyncio.sleep(min(0.2 * (2**attempt), 1.0))

    if last_error is None:
        last_error = RuntimeError("No LLM providers configured")
    raise AllProvidersFailed(last_error, providers)


async def analyze_sentiment(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze transcript sentiment using LLM."""
    if not state.transcript:
        return {"sentiment": {"sentiment": "neutral", "score": 0.5, "emotions": []}}

    response = await _invoke_with_fallback(
        "Analyze the sentiment of this call transcript. "
        "Return ONLY valid JSON: "
        '{"sentiment": "positive"|"neutral"|"negative", '
        '"score": 0.0-1.0, '
        '"emotions": ["emotion1"], '
        '"trend": "improving"|"stable"|"declining", '
        '"alert": true/false}\n\n'
        f"Transcript: {state.transcript[:4000]}",
        state,
        runtime.context,
    )

    try:
        result = json.loads(str(response.content))
    except (json.JSONDecodeError, TypeError):
        result = {
            "sentiment": "neutral",
            "score": 0.5,
            "emotions": [],
            "trend": "stable",
            "alert": False,
        }

    result["call_id"] = state.call_id
    result["tenant_id"] = state.tenant_id
    return {"sentiment": result}


async def generate_coaching(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate real-time coaching tip using LLM."""
    sentiment = state.sentiment or {}
    score = sentiment.get("score", 0.5)
    emotions = sentiment.get("emotions", [])

    response = await _invoke_with_fallback(
        "You are a real-time call coach for a contact center agent. "
        f"Current sentiment: {sentiment.get('sentiment', 'neutral')} (score: {score}). "
        f"Emotions detected: {', '.join(emotions) if emotions else 'none'}. "
        "Provide a brief, actionable coaching tip (1-2 sentences). "
        "Focus on de-escalation, empathy, compliance, or upsell as appropriate.\n\n"
        f"Recent transcript: {state.transcript[:2000]}",
        state,
        runtime.context,
    )

    return {"coaching_tip": response.content}


async def generate_summary(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate post-call summary using LLM."""
    response = await _invoke_with_fallback(
        "Generate a post-call summary. Return ONLY valid JSON: "
        '{"synopsis": "2-3 sentences", '
        '"topics": ["topic1", "topic2"], '
        '"action_items": [{"item": "...", "assignee": "agent|customer"}], '
        '"sentiment_overall": "positive|neutral|negative", '
        '"follow_up_needed": true/false, '
        '"follow_up_reason": "reason or null"}\n\n'
        f"Call ID: {state.call_id}\n"
        f"Duration: {state.duration_seconds}s\n"
        f"Disposition: {state.disposition}\n"
        f"Transcript: {state.transcript[:4000]}",
        state,
        runtime.context,
    )

    try:
        result = json.loads(str(response.content))
    except (json.JSONDecodeError, TypeError):
        result = {"synopsis": response.content, "topics": [], "action_items": []}

    return {"summary": result}


async def score_quality(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Score call quality (rule-based for speed)."""
    t = (state.transcript or "").lower()
    has_greeting = any(w in t for w in ["hello", "hi", "good morning", "welcome"])
    has_empathy = any(w in t for w in ["understand", "sorry", "appreciate"])
    has_resolution = any(w in t for w in ["resolved", "fixed", "taken care"])

    greeting = 8 if has_greeting else 4
    empathy = 8 if has_empathy else 5
    resolution = 9 if has_resolution else 5
    overall = int((greeting + empathy + resolution) / 3 * 10)

    return {
        "qa_score": {
            "overall_score": overall,
            "greeting": greeting,
            "empathy": empathy,
            "resolution": resolution,
            "call_id": state.call_id,
        }
    }


async def score_lead(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Score and qualify lead (rule-based for speed)."""
    sentiment_score = (state.sentiment or {}).get("score", 0.5)
    duration_factor = min(state.duration_seconds / 300, 1.0)
    engagement = sentiment_score * 0.4 + duration_factor * 0.6
    score = int(engagement * 100)

    return {
        "lead_score": {
            "score": score,
            "qualification": "hot" if score > 70 else "warm" if score > 40 else "cold",
            "recommended_action": "schedule_callback" if score > 70 else "nurture",
            "call_id": state.call_id,
        }
    }


async def suggest_routing(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Suggest call routing (rule-based for speed)."""
    t = (state.transcript or "").lower()
    return {
        "routing": {
            "suggested_queue": "retention" if "cancel" in t else "general",
            "priority": 8 if "urgent" in t else 5,
            "reason": "Caller mentioned cancellation"
            if "cancel" in t
            else "Standard routing",
            "call_id": state.call_id,
        }
    }


def route_by_action(state: State) -> str:
    """Route to the correct node based on requested action."""
    action_map = {
        "sentiment": "analyze_sentiment",
        "coaching": "analyze_sentiment",
        "summary": "analyze_sentiment",
        "qa": "score_quality",
        "lead_score": "analyze_sentiment",
        "route": "suggest_routing",
    }
    return action_map.get(state.action, "analyze_sentiment")


def route_after_sentiment(state: State) -> str:
    """Route after sentiment analysis based on original action."""
    if state.action == "coaching":
        return "generate_coaching"
    elif state.action == "summary":
        return "generate_summary"
    elif state.action == "lead_score":
        return "score_lead"
    return END


builder = StateGraph(State, context_schema=Context)

builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("generate_coaching", generate_coaching)
builder.add_node("generate_summary", generate_summary)
builder.add_node("score_quality", score_quality)
builder.add_node("score_lead", score_lead)
builder.add_node("suggest_routing", suggest_routing)

builder.add_conditional_edges("__start__", route_by_action)
builder.add_conditional_edges("analyze_sentiment", route_after_sentiment)

builder.add_edge("generate_coaching", END)
builder.add_edge("generate_summary", END)
builder.add_edge("score_quality", END)
builder.add_edge("score_lead", END)
builder.add_edge("suggest_routing", END)

graph = builder.compile(name="Contact Center Agent")
