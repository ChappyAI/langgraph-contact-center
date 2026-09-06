import importlib
import os

os.environ.setdefault("LANG_API_KEY", "test-key")

import pytest
from fastapi import HTTPException

from agent.graph import AllProvidersFailed, State
from agent.models import get_llm, is_terminal_provider_error

graph = importlib.import_module("agent.graph")


class FakeResponse:
    content = '{"sentiment": "positive", "score": 0.9}'


class FakeLLM:
    def __init__(self, failures=0, error=None):
        self.failures = failures
        self.error = error or RuntimeError("temporary failure")
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return FakeResponse()


@pytest.mark.anyio
async def test_terminal_primary_uses_fallback_once(monkeypatch):
    primary = FakeLLM(error=RuntimeError("no credits remaining"), failures=1)
    fallback = FakeLLM()
    monkeypatch.setattr(
        graph, "get_llm_chain", lambda *args, **kwargs: [primary, fallback]
    )

    result = await graph._invoke_with_fallback("prompt", State(provider="openai"))

    assert result.content.startswith("{")
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.anyio
async def test_retryable_primary_retries_before_success(monkeypatch):
    primary = FakeLLM(failures=2, error=RuntimeError("upstream timeout"))
    monkeypatch.setattr(graph, "get_llm_chain", lambda *args, **kwargs: [primary])
    monkeypatch.setattr(graph.asyncio, "sleep", lambda delay: _completed())

    result = await graph._invoke_with_fallback("prompt", State(provider="openai"))

    assert result.content.startswith("{")
    assert primary.calls == 3


async def _completed():
    return None


def test_provider_classifier_preserves_rate_limits_as_retryable():
    assert is_terminal_provider_error(RuntimeError("no credits remaining"))
    assert is_terminal_provider_error(RuntimeError("invalid_api_key"))
    assert is_terminal_provider_error(RuntimeError("HTTP 403 permission denied"))
    assert not is_terminal_provider_error(RuntimeError("HTTP 429 rate limit"))
    assert not is_terminal_provider_error(RuntimeError("HTTP 503 upstream failure"))


def test_non_openai_provider_ignores_openai_proxy(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = get_llm("anthropic", "claude-test")

    assert client.__class__.__name__ == "ChatAnthropic"


@pytest.mark.anyio
async def test_endpoint_returns_non_2xx_for_all_provider_failure(monkeypatch):
    from api import main

    async def fail(*args, **kwargs):
        raise AllProvidersFailed(
            RuntimeError("no credits remaining"), ["openai", "anthropic"]
        )

    monkeypatch.setattr(main, "_run_graph", fail)
    request = main.InvokeRequest(action="sentiment", call_id="call-1")

    with pytest.raises(HTTPException) as raised:
        await main.invoke(request)

    assert raised.value.status_code == 502
    assert raised.value.detail["error_type"] == "AllProvidersFailed"
