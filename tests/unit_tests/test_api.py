import importlib
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

os.environ.setdefault("LANG_API_KEY", "test-key")

import api.main as api_main

agent_graph = importlib.import_module("agent.graph")


def _context_payload() -> dict[str, object]:
    return {
        "call_id": "call-1",
        "tenant_id": "tenant-1",
        "duration_seconds": 30,
        "lead_id": "lead-1",
        "hangup_cause": "completed",
        "campaign_id": "campaign-1",
        "campaign_outcome": "converted",
    }


def test_call_context_normalizes_hangup_cause_and_preserves_fields() -> None:
    context = api_main._assemble_call_context("summary", _context_payload())

    assert context["disposition"] == "completed"
    assert context["lead_id"] == "lead-1"
    assert context["campaign_id"] == "campaign-1"
    assert context["campaign_outcome"] == "converted"


def test_call_context_rejects_missing_required_fields() -> None:
    payload = _context_payload()
    payload["lead_id"] = None

    with pytest.raises(HTTPException, match="incomplete call context"):
        api_main._assemble_call_context("sentiment", payload)


@pytest.mark.anyio
async def test_run_graph_returns_cached_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class FakeGraph:
        async def ainvoke(
            self, state: object, config: dict[str, object]
        ) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {"sentiment": {"score": 0.9}}

    api_main._RESULT_CACHE.clear()
    monkeypatch.setattr(agent_graph, "graph", FakeGraph())
    payload = {**_context_payload(), "transcript": "hello"}

    first = await api_main._run_graph("sentiment", payload, {})
    second = await api_main._run_graph(
        "sentiment", {**payload, "transcript": "changed"}, {}
    )

    assert calls == 1
    assert second == first


@pytest.mark.anyio
async def test_generate_summary_exposes_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        async def ainvoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(content='{"synopsis": "done"}')

    monkeypatch.setattr(agent_graph, "_llm", lambda: FakeLLM())
    state = agent_graph.State(disposition="completed")

    result = await agent_graph.generate_summary(state, None)

    assert result["summary"]["disposition"] == "completed"
    assert result["summary"]["hangup_cause"] == "completed"
