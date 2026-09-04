"""Con.Nexus Contact Center AI Agent — LangGraph implementation.

Multi-step agent for contact center operations:
- Analyze call transcripts for sentiment and coaching (LLM-powered)
- Score leads from call data
- Generate post-call summaries (LLM-powered)
- Evaluate call quality
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.models import get_llm

COACHING_MAX_TOKENS = 300
SUMMARY_MAX_TOKENS = 600
SUMMARY_RETRY_MAX_TOKENS = 1200
TRUNCATION_REASONS = {"length", "max_tokens"}


@cache
def _llm(max_tokens: int | None = None) -> Any:
    """Lazily construct the LLM so importing this module does not require an API key.

    The OpenAI SDK validates the API key at client construction, so eager
    instantiation at import time breaks any environment (including CI) that
    cannot or does not need to talk to the model — for example, the unit test
    that only verifies the compiled graph is a Pregel instance.
    """
    return get_llm(max_tokens=max_tokens)


def _finish_reason(response: Any) -> str | None:
    """Return the provider completion stop reason."""
    for attribute in ("response_metadata", "additional_kwargs"):
        metadata = getattr(response, attribute, None)
        if not isinstance(metadata, dict):
            continue
        for key in ("finish_reason", "stop_reason"):
            reason = metadata.get(key)
            if isinstance(reason, str):
                return reason.lower()
    return None


def _is_truncated(response: Any) -> bool:
    """Return whether the provider stopped at its completion-token cap."""
    return _finish_reason(response) in TRUNCATION_REASONS


def _valid_summary(value: Any) -> bool:
    """Validate the summary contract returned by the model."""
    if not isinstance(value, dict):
        return False
    required_keys = {
        "synopsis",
        "topics",
        "action_items",
        "sentiment_overall",
        "follow_up_needed",
        "follow_up_reason",
    }
    if not required_keys.issubset(value):
        return False
    if not isinstance(value["synopsis"], str):
        return False
    if not isinstance(value["topics"], list) or not all(
        isinstance(topic, str) for topic in value["topics"]
    ):
        return False
    if not isinstance(value["action_items"], list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("item"), str)
        and isinstance(item.get("assignee"), str)
        for item in value["action_items"]
    ):
        return False
    if value["sentiment_overall"] not in {"positive", "neutral", "negative"}:
        return False
    if not isinstance(value["follow_up_needed"], bool):
        return False
    return value["follow_up_reason"] is None or isinstance(
        value["follow_up_reason"], str
    )


def _valid_coaching_tip(value: Any) -> bool:
    """Validate that a coaching response is complete and concise."""
    if not isinstance(value, str):
        return False
    tip = value.strip()
    if not tip or tip.endswith("...") or tip[-1] not in ".!?":
        return False
    sentences = re.findall(r"[^.!?]+[.!?](?:\s|$)", tip)
    return len(sentences) in {1, 2} and "".join(sentences).strip() == tip


class Context(TypedDict, total=False):
    """Runtime context for the contact center agent."""

    tenant_id: str
    call_id: str
    agent_id: str
    action: str


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

    sentiment: Dict[str, Any] | None = None
    coaching_tip: str | None = None
    summary: Dict[str, Any] | None = None
    qa_score: Dict[str, Any] | None = None
    lead_score: Dict[str, Any] | None = None
    routing: Dict[str, Any] | None = None
    error: str | None = None


async def analyze_sentiment(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze transcript sentiment using LLM."""
    if not state.transcript:
        return {"sentiment": {"sentiment": "neutral", "score": 0.5, "emotions": []}}

    response = await _llm().ainvoke(
        "Analyze the sentiment of this call transcript. "
        "Return ONLY valid JSON: "
        '{"sentiment": "positive"|"neutral"|"negative", '
        '"score": 0.0-1.0, '
        '"emotions": ["emotion1"], '
        '"trend": "improving"|"stable"|"declining", '
        '"alert": true/false}\n\n'
        f"Transcript: {state.transcript[:4000]}"
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

    response = await _llm(max_tokens=COACHING_MAX_TOKENS).ainvoke(
        "You are a real-time call coach for a contact center agent. "
        f"Current sentiment: {sentiment.get('sentiment', 'neutral')} (score: {score}). "
        f"Emotions detected: {', '.join(emotions) if emotions else 'none'}. "
        "Provide a brief, actionable coaching tip (1-2 sentences). "
        "Focus on de-escalation, empathy, compliance, or upsell as appropriate.\n\n"
        f"Recent transcript: {state.transcript[:2000]}"
    )

    if _is_truncated(response) or not _valid_coaching_tip(response.content):
        return {"error": "coaching generation was incomplete"}
    return {"coaching_tip": response.content.strip()}


async def generate_summary(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate post-call summary using LLM."""
    prompt = (
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
        f"Transcript: {state.transcript[:4000]}"
    )
    response = await _llm(max_tokens=SUMMARY_MAX_TOKENS).ainvoke(prompt)
    if _is_truncated(response):
        response = await _llm(max_tokens=SUMMARY_RETRY_MAX_TOKENS).ainvoke(prompt)
        if _is_truncated(response):
            return {"error": "summary generation was truncated"}

    try:
        result = json.loads(str(response.content))
    except (json.JSONDecodeError, TypeError):
        return {"error": "summary generation returned invalid JSON"}

    if not _valid_summary(result):
        return {"error": "summary generation returned an invalid schema"}
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
