import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class CapturedLLM:
    def __init__(self, run_tree):
        self.run_tree = run_tree
        self.model_name = "stub-model"

    async def ainvoke(self, prompt, config=None):
        self.run_tree.append({"run_type": "llm", "inputs": prompt, "config": config})
        return SimpleNamespace(
            content='{"sentiment": "positive", "score": 0.9, "emotions": []}'
        )


@pytest.mark.anyio
async def test_graph_run_captures_root_metadata_and_llm_child(monkeypatch):
    monkeypatch.setenv("LANG_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_ENVIRONMENT", "test")
    main = importlib.import_module("api.main")
    graph_module = importlib.import_module("agent.graph")
    run_tree = []
    llm = CapturedLLM(run_tree)
    monkeypatch.setattr(graph_module, "_llm", lambda: llm)

    async def ainvoke(state, config):
        run_tree.append({"run_type": "chain", "metadata": config["metadata"]})
        await graph_module.analyze_sentiment(state, SimpleNamespace())
        return {"ok": True}

    monkeypatch.setattr(main, "INVOKE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(graph_module.graph, "ainvoke", AsyncMock(side_effect=ainvoke))

    await main._run_graph(
        "sentiment",
        {
            "call_id": "call-123",
            "tenant_id": "tenant-456",
            "trigger": "post_call",
            "provider": "openai",
            "model": "stub-model",
            "transcript": "The caller was happy.",
        },
        {"agent_id": "sentiment-agent"},
    )

    root = run_tree[0]
    assert root["run_type"] == "chain"
    assert root["metadata"] == {
        "call_id": "call-123",
        "tenant_id": "tenant-456",
        "action": "sentiment",
        "trigger": "post_call",
        "provider": "openai",
        "model": "stub-model",
        "thread_id": "call-123",
        "environment": "test",
        "agent_id": "sentiment-agent",
    }
    assert run_tree[1]["run_type"] == "llm"
    assert run_tree[1]["config"]["metadata"]["provider"] == "openai"
    assert run_tree[1]["config"]["metadata"]["model"] == "stub-model"
