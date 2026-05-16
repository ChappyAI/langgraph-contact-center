# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LangGraph state machine for the Chappy Connect (Con.Nexus) contact center. Single compiled graph (`Contact Center Agent`) dispatches one of six action paths against a call transcript: sentiment analysis, real-time coaching, post-call summary, QA scoring, lead scoring, and routing suggestion. Sentiment/coaching/summary use `gpt-4o-mini` via `langchain-openai`; QA, lead score, and routing are deterministic rule-based nodes.

The wider repo (`telephony-service`, `telephony-service-frontend`) calls this graph through `LANGGRAPH_API_URL` when `FEATURE_EXTERNAL_AGENCY` is enabled — see the parent `CLAUDE.md` at `/Users/seanchapman/DDev/containers/telephony/luhx/CLAUDE.md` for the wider system map.

## Package manager: uv

The project ships a `uv.lock`. Use `uv`, not bare `pip`. A stale `VIRTUAL_ENV` env var pointing at `~/telephony/Alltalkpro/langgraph-app/.venv` is sometimes inherited from the parent shell — `unset VIRTUAL_ENV` before activating `.venv` here, or use `uv run` which selects the project venv automatically.

```bash
uv sync                              # install from uv.lock (incl. dev group)
uv run langgraph dev                 # start LangGraph dev server + Studio
uv run python -m pytest              # run all tests
uv run pytest tests/unit_tests/test_configuration.py::test_placeholder
uv run pytest tests/integration_tests # integration only (hits OpenAI)
uv run ruff check . && uv run ruff format .
uv run mypy --strict src/
```

The `Makefile` targets (`make test`, `make lint`, `make format`) assume the venv is already activated and call bare `python -m pytest` / `ruff` / `mypy`. They work, but `uv run` is the safer entry point when running from a fresh shell.

## Architecture

Single file: **`src/agent/graph.py`** defines everything. `langgraph.json` points `graphs.agent` → `./src/agent/graph.py:graph`, which is how the LangGraph CLI/Studio discovers the compiled graph.

Two type contracts live alongside the nodes:

- `Context` (TypedDict) — runtime context the caller passes in: `tenant_id`, `call_id`, `agent_id`, `action`. This is what `runtime: Runtime[Context]` exposes inside each node.
- `State` (dataclass) — the message flowing through the graph. Includes both inputs (`transcript`, `disposition`, `duration_seconds`, `caller_number`, …) and per-node outputs (`sentiment`, `coaching_tip`, `summary`, `qa_score`, `lead_score`, `routing`). Nodes return a partial dict; LangGraph merges it into `State`.

### Dispatch flow

The entrypoint is a conditional edge from `__start__` driven by `state.action`:

```
__start__ ──route_by_action(state.action)──► one of {analyze_sentiment, score_quality, suggest_routing}
analyze_sentiment ──route_after_sentiment──► {generate_coaching, generate_summary, score_lead, END}
{score_quality, suggest_routing, generate_coaching, generate_summary, score_lead} ──► END
```

Three of the six actions (`coaching`, `summary`, `lead_score`) intentionally hop through `analyze_sentiment` first because the downstream nodes consume `state.sentiment`. Don't shortcut the path — coaching reads `sentiment.emotions`, lead scoring weights `sentiment.score` 40% of the engagement signal.

### LLM call shape

Sentiment and summary nodes ask the LLM for **JSON only**, then `json.loads` the response inside a `try/except (JSONDecodeError, TypeError)` that falls back to a safe default dict. Don't tighten this into a hard failure — the graph runs against live calls and a bad LLM token should not break the trace. Truncations: transcript clipped to 4000 chars for sentiment/summary, 2000 for coaching.

`llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)` is module-level. Tests that hit `analyze_sentiment`, `generate_coaching`, or `generate_summary` need `OPENAI_API_KEY` in the environment (loaded from `.env` by LangGraph CLI; pytest does not auto-load it).

### Rule-based scoring

`score_quality`, `score_lead`, `suggest_routing` are deliberately keyword/heuristic-based for latency. They run inline without an LLM. Keep them cheap — they are the fallback path when the LLM is rate-limited or disabled.

## Tests

- `tests/unit_tests/test_configuration.py` — only verifies the graph compiles to a `Pregel` instance. Cheap smoke test, no API calls.
- `tests/integration_tests/test_graph.py` — `@pytest.mark.langsmith` + `@pytest.mark.anyio`, invokes `graph.ainvoke(...)` against the real LLM. Requires `OPENAI_API_KEY` and (for the LangSmith decorator) `LANGSMITH_API_KEY`.
- `tests/conftest.py` configures anyio. `pytestmark = pytest.mark.anyio` at the module level is required for async tests.

## Conventions

- Python 3.10+ (`requires-python = ">=3.10"`). `from __future__ import annotations` at the top of `graph.py` keeps annotations as strings — preserve this.
- Ruff is the formatter and linter (pydocstyle google convention, imperative first lines via `D401`). `mypy --strict` is enforced by `make lint`.
- The setuptools config exposes the package under **two** names: `agent` and `langgraph.templates.agent`, both pointing at `src/agent`. Don't rename the import path; the LangGraph platform expects the template namespace.
