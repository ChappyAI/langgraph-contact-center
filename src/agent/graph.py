"""AllTalkPro Contact Center AI Agent — LangGraph implementation.

Multi-step agent for contact center operations:
- Analyze call transcripts for sentiment and coaching (LLM-powered)
- Score leads from call data
- Generate post-call summaries (LLM-powered)
- Evaluate call quality
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


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

    sentiment: Optional[Dict[str, Any]] = None
    coaching_tip: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    qa_score: Optional[Dict[str, Any]] = None
    lead_score: Optional[Dict[str, Any]] = None
    routing: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


async def analyze_sentiment(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze transcript sentiment using LLM."""
    if not state.transcript:
        return {"sentiment": {"sentiment": "neutral", "score": 0.5, "emotions": []}}

    response = await llm.ainvoke(
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
        result = json.loads(response.content)
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

    response = await llm.ainvoke(
        "You are a real-time call coach for a contact center agent. "
        f"Current sentiment: {sentiment.get('sentiment', 'neutral')} (score: {score}). "
        f"Emotions detected: {', '.join(emotions) if emotions else 'none'}. "
        "Provide a brief, actionable coaching tip (1-2 sentences). "
        "Focus on de-escalation, empathy, compliance, or upsell as appropriate.\n\n"
        f"Recent transcript: {state.transcript[:2000]}"
    )

    return {"coaching_tip": response.content}


async def generate_summary(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate post-call summary using LLM."""
    response = await llm.ainvoke(
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

    try:
        result = json.loads(response.content)
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
            "reason": "Caller mentioned cancellation" if "cancel" in t else "Standard routing",
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
