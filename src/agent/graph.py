"""AllTalkPro Contact Center AI Agent — LangGraph implementation.

Multi-step agent for contact center operations:
- Analyze call transcripts for sentiment and coaching
- Score leads from call data
- Generate post-call summaries
- Evaluate call quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class Context(TypedDict, total=False):
    """Runtime context for the contact center agent."""
    tenant_id: str
    call_id: str
    agent_id: str
    action: str  # "sentiment" | "coaching" | "summary" | "qa" | "lead_score" | "route"


@dataclass
class State:
    """State flowing through the contact center agent graph."""
    # Input
    action: str = "sentiment"
    transcript: str = ""
    call_id: str = ""
    tenant_id: str = ""
    agent_id: str = ""
    duration_seconds: int = 0
    disposition: str = ""
    caller_number: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Output (accumulated by nodes)
    sentiment: Optional[Dict[str, Any]] = None
    coaching_tip: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    qa_score: Optional[Dict[str, Any]] = None
    lead_score: Optional[Dict[str, Any]] = None
    routing: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


async def analyze_sentiment(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze transcript sentiment."""
    if not state.transcript:
        return {"sentiment": {"sentiment": "neutral", "score": 0.5, "emotions": []}}

    return {
        "sentiment": {
            "sentiment": "negative" if any(w in state.transcript.lower() for w in ["frustrated", "angry", "cancel", "terrible"]) else
                        "positive" if any(w in state.transcript.lower() for w in ["thank", "great", "happy", "excellent"]) else
                        "neutral",
            "score": 0.3 if any(w in state.transcript.lower() for w in ["frustrated", "angry", "cancel"]) else
                     0.8 if any(w in state.transcript.lower() for w in ["thank", "great", "happy"]) else 0.5,
            "emotions": [w for w in ["frustration", "anger", "satisfaction", "gratitude"]
                        if w[:4] in state.transcript.lower()],
            "call_id": state.call_id,
            "tenant_id": state.tenant_id,
        }
    }


async def generate_coaching(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate real-time coaching tip based on sentiment."""
    sentiment = state.sentiment or {}
    score = sentiment.get("score", 0.5)

    if score < 0.4:
        tip = "Acknowledge the customer's frustration. Use empathetic language: 'I understand how frustrating this must be.'"
    elif score > 0.7:
        tip = "Great rapport! Consider an upsell opportunity or ask for referral."
    else:
        tip = "Stay engaged. Ask open-ended questions to understand the customer's needs better."

    return {"coaching_tip": tip}


async def generate_summary(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate post-call summary."""
    return {
        "summary": {
            "synopsis": f"Call {state.call_id} lasted {state.duration_seconds}s. Disposition: {state.disposition}.",
            "topics": [w for w in state.transcript.split() if len(w) > 6][:5] if state.transcript else [],
            "action_items": [],
            "sentiment_overall": state.sentiment.get("sentiment", "neutral") if state.sentiment else "neutral",
            "follow_up_needed": state.disposition in ["callback", "escalation", "pending"],
        }
    }


async def score_quality(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Score call quality."""
    has_greeting = any(w in (state.transcript or "").lower() for w in ["hello", "hi", "good morning", "welcome"])
    has_empathy = any(w in (state.transcript or "").lower() for w in ["understand", "sorry", "appreciate"])
    has_resolution = any(w in (state.transcript or "").lower() for w in ["resolved", "fixed", "taken care"])

    greeting_score = 8 if has_greeting else 4
    empathy_score = 8 if has_empathy else 5
    resolution_score = 9 if has_resolution else 5

    overall = int((greeting_score + empathy_score + resolution_score) / 3 * 10)

    return {
        "qa_score": {
            "overall_score": overall,
            "greeting": greeting_score,
            "empathy": empathy_score,
            "resolution": resolution_score,
            "call_id": state.call_id,
        }
    }


async def score_lead(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Score and qualify lead."""
    sentiment_score = (state.sentiment or {}).get("score", 0.5)
    duration_factor = min(state.duration_seconds / 300, 1.0)
    engagement = sentiment_score * 0.4 + duration_factor * 0.6

    return {
        "lead_score": {
            "score": int(engagement * 100),
            "qualification": "hot" if engagement > 0.7 else "warm" if engagement > 0.4 else "cold",
            "recommended_action": "schedule_callback" if engagement > 0.7 else "nurture",
            "call_id": state.call_id,
        }
    }


async def suggest_routing(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Suggest call routing based on context."""
    return {
        "routing": {
            "suggested_queue": "retention" if "cancel" in (state.transcript or "").lower() else "general",
            "priority": 8 if "urgent" in (state.transcript or "").lower() else 5,
            "reason": "Caller mentioned cancellation" if "cancel" in (state.transcript or "").lower() else "Standard routing",
            "call_id": state.call_id,
        }
    }


def route_by_action(state: State) -> str:
    """Route to the correct node based on requested action."""
    action_map = {
        "sentiment": "analyze_sentiment",
        "coaching": "analyze_sentiment",  # coaching needs sentiment first
        "summary": "analyze_sentiment",   # summary needs sentiment first
        "qa": "score_quality",
        "lead_score": "analyze_sentiment",  # lead score needs sentiment first
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


# Build the graph
builder = StateGraph(State, context_schema=Context)

# Add nodes
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("generate_coaching", generate_coaching)
builder.add_node("generate_summary", generate_summary)
builder.add_node("score_quality", score_quality)
builder.add_node("score_lead", score_lead)
builder.add_node("suggest_routing", suggest_routing)

# Conditional entry based on action
builder.add_conditional_edges("__start__", route_by_action)

# After sentiment, route to next step or end
builder.add_conditional_edges("analyze_sentiment", route_after_sentiment)

# Terminal nodes
builder.add_edge("generate_coaching", END)
builder.add_edge("generate_summary", END)
builder.add_edge("score_quality", END)
builder.add_edge("score_lead", END)
builder.add_edge("suggest_routing", END)

graph = builder.compile(name="Contact Center Agent")
