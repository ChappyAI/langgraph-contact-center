from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent import graph


def _client() -> TestClient:
    import os

    os.environ.setdefault("LANG_API_KEY", "test-key")
    from api.main import app

    return TestClient(app)


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["sentiment", "coaching", "summary", "lead_score"])
async def test_empty_transcript_returns_error_without_llm(monkeypatch, action):
    llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=AssertionError("LLM called")))
    monkeypatch.setattr(graph, "_llm", lambda: llm)

    result = await graph.graph.ainvoke(graph.State(action=action, transcript=" \t\n"))

    assert result["error"] == "insufficient_input: transcript is empty"
    llm.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_node_guards_reject_empty_transcripts(monkeypatch):
    llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=AssertionError("LLM called")))
    monkeypatch.setattr(graph, "_llm", lambda: llm)
    state = graph.State(transcript="")

    assert (await graph.analyze_sentiment(state, None))["error"]
    assert (await graph.generate_coaching(state, None))["error"]
    assert (await graph.generate_summary(state, None))["error"]
    llm.ainvoke.assert_not_awaited()


def test_invoke_missing_transcript_returns_422():
    client = _client()

    response = client.post(
        "/invoke",
        headers={"x-api-key": "test-key"},
        json={"action": "sentiment"},
    )

    assert response.status_code == 422
    assert "transcript" in response.text


def test_invoke_empty_transcript_reaches_graph(monkeypatch):
    async def fake_run_graph(action, payload, context):
        assert payload["transcript"] == ""
        return {"error": "insufficient_input: transcript is empty"}

    monkeypatch.setattr("api.main._run_graph", fake_run_graph)
    client = _client()

    response = client.post(
        "/invoke",
        headers={"x-api-key": "test-key"},
        json={"action": "sentiment", "transcript": "   "},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["error"] == "insufficient_input: transcript is empty"


def test_stream_missing_transcript_emits_error_event(monkeypatch):
    run_graph = AsyncMock(side_effect=AssertionError("graph called"))
    monkeypatch.setattr("api.main._run_graph", run_graph)
    client = _client()

    response = client.post(
        "/runs/stream",
        headers={"x-api-key": "test-key"},
        json={"assistant_id": "summary", "input": {}},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "transcript is required" in response.text
    assert "event: values" not in response.text
    run_graph.assert_not_awaited()


@pytest.mark.anyio
async def test_unterminated_coaching_output_is_rejected(monkeypatch):
    llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="Offer empathy"))
    )
    monkeypatch.setattr(graph, "_llm", lambda: llm)

    result = await graph.generate_coaching(
        graph.State(transcript="Caller needs help"), None
    )

    assert result == {"error": "invalid_coaching_output"}
