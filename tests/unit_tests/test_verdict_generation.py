from importlib import import_module
from types import SimpleNamespace

import pytest

graph = import_module("agent.graph")

pytestmark = pytest.mark.anyio


class FakeLLM:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)

    async def ainvoke(self, prompt: str) -> object:
        return self.responses.pop(0)


def response(content: object, **metadata: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, response_metadata=metadata)


async def test_summary_rejects_truncated_json(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM(response('{"synopsis": "cut off'))
    monkeypatch.setattr(graph, "_llm", lambda max_tokens=None: llm)

    result = await graph.generate_summary(graph.State(transcript="A call"), None)

    assert result == {"error": "summary generation returned invalid JSON"}


async def test_summary_rejects_repeated_length_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM(
        response('{"synopsis": "cut off', finish_reason="length"),
        response('{"synopsis": "still cut off', finish_reason="length"),
    )
    monkeypatch.setattr(graph, "_llm", lambda max_tokens=None: llm)

    result = await graph.generate_summary(graph.State(transcript="A call"), None)

    assert result == {"error": "summary generation was truncated"}


async def test_coaching_rejects_mid_word_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM(response("Acknowledge the customer's concern and offer to help res"))
    monkeypatch.setattr(graph, "_llm", lambda max_tokens=None: llm)

    result = await graph.generate_coaching(graph.State(), None)

    assert result == {"error": "coaching generation was incomplete"}
