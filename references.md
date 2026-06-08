# langgraph-contact-center — Codebase Reference

> Comprehensive reference for the LangGraph implementation of the Con.Nexus Contact Center AI agent.
> Generated 2026-05-15. Repository: `chappyai/langgraph-contact-center`.

---

## 1. Purpose & Scope

`langgraph-contact-center` (a.k.a. `langgraph-app` inside the parent `telephony-service-bundle` monorepo) is a Python [LangGraph](https://github.com/langchain-ai/langgraph) application that hosts the **Con.Nexus Contact Center AI Agent**.

It exposes a single compiled graph named **"Contact Center Agent"** (declared as `agent` in `langgraph.json`) that performs the following multi-step contact-center operations on call data:

| Action | Description | Implementation |
| --- | --- | --- |
| `sentiment` | Analyze the sentiment, emotions, and trend of a call transcript. | LLM (OpenAI `gpt-4o-mini`) returning structured JSON. |
| `coaching` | Generate a real-time, 1-2 sentence coaching tip for the live agent based on current sentiment. | LLM prompt, free-form text output. Runs after `analyze_sentiment`. |
| `summary` | Generate a post-call summary, topics, and action items. | LLM returning structured JSON. Runs after `analyze_sentiment`. |
| `qa` | Score call quality on greeting, empathy, and resolution. | Rule-based keyword heuristic, no LLM call. |
| `lead_score` | Score and qualify the caller as `hot`/`warm`/`cold`. | Rule-based, but consumes the sentiment produced by `analyze_sentiment`. |
| `route` | Suggest queue and priority for routing the call. | Rule-based keyword heuristic, no LLM call. |

The graph is designed to be invoked by the NestJS backend in the broader Chappy Connect telephony platform (`telephony-service` / `con-nexus-telephony`) via the LangGraph Server HTTP API, gated by the `FEATURE_EXTERNAL_AGENCY` flag and configured through `LANGGRAPH_API_URL`. See sections 10 and 11 for invocation details.

This is intentionally a small, single-graph project; the surface area is one Python module plus tests and CI plumbing. It originated from the official LangGraph starter template (`new-langgraph-project`) and retains some of that template's scaffolding.

---

## 2. Tech Stack

- **Language / Runtime**: Python `>=3.10` (CI matrix uses 3.11 and 3.12).
- **Core framework**: `langgraph >= 1.0.0` (StateGraph API with `Runtime[Context]` typed context).
- **LLM integration**: `langchain-openai >= 0.3.0` — `ChatOpenAI` bound to `gpt-4o-mini` at `temperature=0.3`.
- **Env handling**: `python-dotenv >= 1.0.1` (transitively, via LangGraph CLI).
- **Dev / packaging**:
  - `uv` (used in CI for venv + dependency resolution; `uv.lock` is committed).
  - `setuptools >= 73`, `wheel` (build backend).
  - `langgraph-cli[inmem] >= 0.4.14` — local server.
  - `pytest >= 8.3.5`, `anyio >= 4.7.0`.
  - `ruff >= 0.8.2`, `mypy >= 1.13.0` (strict), `codespell`.
- **Container base** (LangGraph Cloud): `wolfi` (declared via `image_distro` in `langgraph.json`).

The project ships an `MIT` license (`LICENSE`).

---

## 3. Repository Layout (annotated tree)

```
langgraph-contact-center/
├── .codespellignore              # codespell ignore list (empty)
├── .env.example                  # template for .env (LANGSMITH_PROJECT placeholder)
├── .github/
│   └── workflows/
│       ├── integration-tests.yml # daily cron + manual integration test workflow
│       └── unit-tests.yml        # PR/push CI: ruff, mypy --strict, codespell, pytest
├── .gitignore                    # ignores .env, venv artifacts, .langgraph_api/, .claude/, etc.
├── INDEX.md                      # short orientation doc (predates rename to contact-center)
├── LICENSE                       # MIT
├── Makefile                      # test / lint / format / spell targets
├── README.md                     # original new-langgraph-project starter README
├── langgraph.json                # LangGraph Server config: graph map, env, deps, distro
├── pyproject.toml                # project metadata, deps, ruff/mypy config
├── src/
│   └── agent/
│       ├── __init__.py           # exports `graph` symbol from `agent.graph`
│       └── graph.py              # ALL graph logic: state, nodes, routing, compiled graph
├── static/
│   └── studio_ui.png             # screenshot referenced from README.md
├── tests/
│   ├── conftest.py               # pytest session-scoped `anyio_backend` fixture (asyncio)
│   ├── integration_tests/
│   │   ├── __init__.py
│   │   └── test_graph.py         # async ainvoke smoke test (marked @pytest.mark.langsmith)
│   └── unit_tests/
│       ├── __init__.py
│       └── test_configuration.py # asserts compiled `graph` is a Pregel instance
└── uv.lock                       # uv-resolved dependency lock (~500 KB)
```

Notable observations:
- The `pyproject.toml` `name` is **`agent`** (not `langgraph-contact-center`) and the README still references the starter template — see Section 12.
- `setuptools.packages` registers the source dir under both `agent` and `langgraph.templates.agent`, which is the LangGraph CLI convention for templating discovery.

---

## 4. Graph Definition (`src/agent/graph.py`)

All graph logic lives in a single 226-line module. The shape is:

1. Module-level `ChatOpenAI` client.
2. `Context` TypedDict (runtime context schema for `Runtime[Context]`).
3. `State` dataclass (graph state).
4. Six async node functions.
5. Two routing functions.
6. `StateGraph` builder wiring, then `graph = builder.compile(name="Contact Center Agent")`.

### 4.1 LLM binding

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
```

A single shared client is reused across all LLM-using nodes. `OPENAI_API_KEY` is read from the environment via `langchain-openai`'s default. Temperature `0.3` favors deterministic structured-JSON output without being fully greedy.

### 4.2 State schema

Two schemas are declared:

#### `Context` (runtime context, `TypedDict`, `total=False`)

| Field | Type | Purpose |
| --- | --- | --- |
| `tenant_id` | `str` | Multi-tenant identifier (passed through to result payloads). |
| `call_id` | `str` | Call identifier. |
| `agent_id` | `str` | Live agent identifier. |
| `action` | `str` | Requested action; mirrored on `State.action`. |

`Context` is wired into the graph via `StateGraph(State, context_schema=Context)` and reaches nodes through `runtime: Runtime[Context]`. Currently the node implementations do not read from `runtime`; routing is driven by `State.action`. The `Context` schema is therefore a documented extension point, not an active control surface.

#### `State` (graph state, `@dataclass`)

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `action` | `str` | `"sentiment"` | Drives conditional routing. One of `sentiment`, `coaching`, `summary`, `qa`, `lead_score`, `route`. |
| `transcript` | `str` | `""` | Call transcript text. Truncated to 4000 chars (sentiment/summary) or 2000 chars (coaching) when sent to the LLM. |
| `call_id` | `str` | `""` | Echoed into every result for correlation. |
| `tenant_id` | `str` | `""` | Echoed into `sentiment` result. |
| `agent_id` | `str` | `""` | Currently captured but unused inside nodes. |
| `duration_seconds` | `int` | `0` | Feeds the lead-score engagement formula and is included in the summary prompt. |
| `disposition` | `str` | `""` | Included in the summary prompt. |
| `caller_number` | `str` | `""` | Captured but unused inside nodes (extension point). |
| `metadata` | `Dict[str, Any]` | `{}` | Free-form passthrough metadata. |
| `sentiment` | `Optional[Dict[str, Any]]` | `None` | Output of `analyze_sentiment`. Consumed by `generate_coaching` and `score_lead`. |
| `coaching_tip` | `Optional[str]` | `None` | Output of `generate_coaching`. |
| `summary` | `Optional[Dict[str, Any]]` | `None` | Output of `generate_summary`. |
| `qa_score` | `Optional[Dict[str, Any]]` | `None` | Output of `score_quality`. |
| `lead_score` | `Optional[Dict[str, Any]]` | `None` | Output of `score_lead`. |
| `routing` | `Optional[Dict[str, Any]]` | `None` | Output of `suggest_routing`. |
| `error` | `Optional[str]` | `None` | Reserved for error surfacing (not currently written by any node). |

Each node returns a partial dict whose keys correspond to the state fields above; LangGraph merges those into the running `State` instance.

### 4.3 Nodes

All node functions are `async def` and share the signature
`async def NAME(state: State, runtime: Runtime[Context]) -> Dict[str, Any]`.

#### 4.3.1 `analyze_sentiment`

- **Purpose**: Produce a structured sentiment record for the transcript.
- **Short-circuit**: If `state.transcript` is empty, returns a fixed neutral record `{"sentiment": {"sentiment": "neutral", "score": 0.5, "emotions": []}}` and does **not** call the LLM. Note that in this branch the returned object is missing `trend`/`alert`/`call_id`/`tenant_id` fields that downstream consumers may expect — see Section 12.
- **Prompt** (truncated transcript to 4000 chars):

  > "Analyze the sentiment of this call transcript. Return ONLY valid JSON: `{"sentiment": "positive"|"neutral"|"negative", "score": 0.0-1.0, "emotions": ["emotion1"], "trend": "improving"|"stable"|"declining", "alert": true/false}`"

- **Output (success)**: a JSON object parsed from `response.content`, then augmented with `call_id` and `tenant_id` from `State`. On `json.JSONDecodeError` / `TypeError`, falls back to `{sentiment: "neutral", score: 0.5, emotions: [], trend: "stable", alert: False}`.
- **Returns**: `{"sentiment": <result>}`.

#### 4.3.2 `generate_coaching`

- **Purpose**: Real-time coaching tip for the live agent.
- **Reads** `state.sentiment` (typically populated by the prior `analyze_sentiment` run). Pulls `sentiment.sentiment`, `sentiment.score`, `sentiment.emotions` (each with safe defaults).
- **Prompt** (truncated transcript to 2000 chars):

  > "You are a real-time call coach for a contact center agent. Current sentiment: {label} (score: {score}). Emotions detected: {emotions}. Provide a brief, actionable coaching tip (1-2 sentences). Focus on de-escalation, empathy, compliance, or upsell as appropriate."

- **Returns**: `{"coaching_tip": response.content}` — raw LLM text (not JSON).

#### 4.3.3 `generate_summary`

- **Purpose**: Post-call structured summary.
- **Prompt inputs**: `call_id`, `duration_seconds`, `disposition`, transcript truncated to 4000 chars.
- **Expected JSON shape**:
  ```json
  {
    "synopsis": "2-3 sentences",
    "topics": ["topic1", "topic2"],
    "action_items": [{"item": "...", "assignee": "agent|customer"}],
    "sentiment_overall": "positive|neutral|negative",
    "follow_up_needed": true,
    "follow_up_reason": "reason or null"
  }
  ```
- **Fallback** on parse failure: `{"synopsis": <raw content>, "topics": [], "action_items": []}`.
- **Returns**: `{"summary": <result>}`.

#### 4.3.4 `score_quality`

- **Purpose**: Quick QA scoring without an LLM call.
- **Algorithm** (case-insensitive keyword presence):
  - `has_greeting` → any of `hello`, `hi`, `good morning`, `welcome` → `greeting = 8` else `4`.
  - `has_empathy` → any of `understand`, `sorry`, `appreciate` → `empathy = 8` else `5`.
  - `has_resolution` → any of `resolved`, `fixed`, `taken care` → `resolution = 9` else `5`.
  - `overall = int((greeting + empathy + resolution) / 3 * 10)`.
- **Returns**: `{"qa_score": {"overall_score": overall, "greeting": ..., "empathy": ..., "resolution": ..., "call_id": ...}}`.
- **Note**: `overall` is an integer on a 0–100 scale (the sub-scores are 0–10 themselves), so a "perfect" call yields `83`. See Section 12.

#### 4.3.5 `score_lead`

- **Purpose**: Lead scoring / qualification combining sentiment and duration.
- **Algorithm**:
  - `sentiment_score = state.sentiment["score"]` (default `0.5`).
  - `duration_factor = min(duration_seconds / 300, 1.0)` — saturates at 5 minutes.
  - `engagement = 0.4 * sentiment_score + 0.6 * duration_factor`.
  - `score = int(engagement * 100)`.
  - Qualification: `hot` if `> 70`, else `warm` if `> 40`, else `cold`.
  - Recommended action: `schedule_callback` if `> 70`, else `nurture`.
- **Returns**: `{"lead_score": {"score": ..., "qualification": ..., "recommended_action": ..., "call_id": ...}}`.

#### 4.3.6 `suggest_routing`

- **Purpose**: Heuristic queue/priority suggestion.
- **Algorithm** (on lowercased transcript):
  - `suggested_queue = "retention"` if `"cancel" in transcript` else `"general"`.
  - `priority = 8` if `"urgent" in transcript` else `5`.
  - `reason = "Caller mentioned cancellation"` if `"cancel"` else `"Standard routing"`.
- **Returns**: `{"routing": {"suggested_queue": ..., "priority": ..., "reason": ..., "call_id": ...}}`.

### 4.4 Routing functions

#### 4.4.1 `route_by_action(state)`

Called from the synthetic `__start__` node via `add_conditional_edges("__start__", route_by_action)`. Maps `state.action` to the first node to execute:

```python
action_map = {
    "sentiment":   "analyze_sentiment",
    "coaching":    "analyze_sentiment",   # needs sentiment first
    "summary":     "analyze_sentiment",   # currently also gated through sentiment
    "qa":          "score_quality",
    "lead_score":  "analyze_sentiment",   # needs sentiment first
    "route":       "suggest_routing",
}
return action_map.get(state.action, "analyze_sentiment")  # default = sentiment
```

Implications:
- **Three of six actions** (`coaching`, `summary`, `lead_score`) are routed through `analyze_sentiment` first, so they always incur the sentiment LLM call before their own work.
- `summary` *does not strictly require* sentiment; it is sent through `analyze_sentiment` anyway. See Section 12.
- Unknown actions silently default to `sentiment`.

#### 4.4.2 `route_after_sentiment(state)`

Called from `analyze_sentiment` via `add_conditional_edges("analyze_sentiment", route_after_sentiment)`:

```python
if state.action == "coaching":   return "generate_coaching"
elif state.action == "summary":  return "generate_summary"
elif state.action == "lead_score": return "score_lead"
return END
```

So when the user requested `action == "sentiment"` (or anything unmapped), the graph terminates immediately after `analyze_sentiment`.

### 4.5 Edges & overall topology

Static edges added with `add_edge(...)`:

| From | To |
| --- | --- |
| `generate_coaching` | `END` |
| `generate_summary` | `END` |
| `score_quality` | `END` |
| `score_lead` | `END` |
| `suggest_routing` | `END` |

Conditional edges:

| From | Router | Possible destinations |
| --- | --- | --- |
| `__start__` | `route_by_action` | `analyze_sentiment`, `score_quality`, `suggest_routing` |
| `analyze_sentiment` | `route_after_sentiment` | `generate_coaching`, `generate_summary`, `score_lead`, `END` |

Visually:

```
__start__
  ├── action ∈ {sentiment, coaching, summary, lead_score, *unknown*} ──► analyze_sentiment
  │       ├── action == coaching   ──► generate_coaching ──► END
  │       ├── action == summary    ──► generate_summary  ──► END
  │       ├── action == lead_score ──► score_lead        ──► END
  │       └── otherwise            ──► END
  ├── action == qa                                       ──► score_quality   ──► END
  └── action == route                                    ──► suggest_routing ──► END
```

### 4.6 Compiled graph entry point

```python
builder = StateGraph(State, context_schema=Context)
# add_node x6, add_conditional_edges x2, add_edge x5
graph = builder.compile(name="Contact Center Agent")
```

- Module-level symbol exported as `graph` from `src/agent/__init__.py` (`from agent.graph import graph`).
- This symbol is what `langgraph.json` references (`./src/agent/graph.py:graph`).
- No checkpointer is configured at compile time — persistence is delegated to the LangGraph server / Cloud (which injects an in-memory checkpointer for `langgraph dev` and Postgres for the platform deployment).

---

## 5. Configuration (`langgraph.json`)

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "image_distro": "wolfi"
}
```

- **`dependencies: ["."]`** — installs the current package; combined with `setuptools` configuration in `pyproject.toml`, this exposes the `agent` import.
- **`graphs.agent`** — the graph is published under the key `"agent"` on the LangGraph server. Clients select it via `assistant_id="agent"` (LangGraph auto-creates a default assistant per graph id).
- **`env: ".env"`** — `.env` is loaded by `langgraph dev` / `langgraph up` at process start. The `.gitignore` excludes `.env` from version control.
- **`image_distro: "wolfi"`** — selects the Chainguard Wolfi-based Python base image when building for LangGraph Cloud / Platform.

No `pip_config_file`, `dockerfile_lines`, or `auth` blocks are configured; the default LangGraph Server behavior applies (no auth on local `langgraph dev`).

---

## 6. Environment Variables

There is no central environment variable schema in the code; values are consumed by the LangGraph / LangChain libraries directly. The variables of practical interest are:

| Variable | Source | Required? | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | `langchain-openai` `ChatOpenAI` default. | **Yes** for any LLM-using action (`sentiment`, `coaching`, `summary`, and the three actions that fan through `analyze_sentiment`). | Authenticates `gpt-4o-mini` calls. |
| `OPENAI_BASE_URL` | `langchain-openai`. | No | Override OpenAI endpoint (proxy / Azure-style routing). |
| `LANGSMITH_API_KEY` | LangSmith / LangChain tracing. Referenced in CI integration job (`secrets.LANGSMITH_API_KEY`). | No | Enables LangSmith tracing of runs. |
| `LANGSMITH_PROJECT` | `.env.example` ships `LANGSMITH_PROJECT=new-agent`. | No | Project name in LangSmith. |
| `LANGSMITH_TRACING` | Set to `true` in the integration test workflow. | No | Master switch for LangSmith tracing. |
| `ANTHROPIC_API_KEY` | Referenced in `integration-tests.yml` only (legacy from the starter template; not consumed by current code). | No | Vestigial — see Section 12. |

`.env` itself is git-ignored. `.env.example` contains only:

```
LANGSMITH_PROJECT=new-agent
```

Any additional secrets must be supplied through `.env` (local) or platform secret stores (LangGraph Cloud / Docker / k8s).

---

## 7. Tools / Integrations

This graph has no LangChain `Tool` objects, no MCP integrations, no retrievers, and no external HTTP clients. The full integration surface is:

- **OpenAI Chat Completions** via `langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0.3)`. Called from three nodes (`analyze_sentiment`, `generate_coaching`, `generate_summary`) using `await llm.ainvoke(prompt: str)` with a plain string prompt (no chat-message list, no function-calling, no structured-output `with_structured_output(...)` binding — JSON contract is enforced solely through prompt text and `json.loads` with a try/except fallback).
- **LangGraph Runtime**: each node receives a `Runtime[Context]` parameter, but the body never reads it. The `Context` TypedDict is wired through for future use (or for client-side metadata propagation through the LangGraph server).
- **LangSmith**: optional, controlled purely by environment variables (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`). When set, `langchain` will automatically trace every `ainvoke`.

Rule-based nodes (`score_quality`, `score_lead`, `suggest_routing`) call no external services.

---

## 8. Tests

The `tests/` package is small. `tests/conftest.py` declares one session-scoped fixture used by `pytest.mark.anyio`:

```python
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
```

### 8.1 Unit tests — `tests/unit_tests/test_configuration.py`

```python
from langgraph.pregel import Pregel
from agent.graph import graph

def test_placeholder() -> None:
    assert isinstance(graph, Pregel)
```

This is the **only** unit test. It verifies that import-time graph construction succeeds and that `builder.compile(...)` produces a Pregel object. There are no tests for individual node functions, routing, or LLM-shape contracts. See Section 12 for gaps.

### 8.2 Integration tests — `tests/integration_tests/test_graph.py`

```python
import pytest
from agent import graph

pytestmark = pytest.mark.anyio

@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {"changeme": "some_val"}
    res = await graph.ainvoke(inputs)
    assert res is not None
```

Notes:
- The input `{"changeme": "some_val"}` is **not** a `State` field. Since `State` is a dataclass with defaults, LangGraph will ignore the unknown key and start from defaults (`action="sentiment"`, empty transcript). With an empty transcript, `analyze_sentiment` short-circuits to a neutral payload without calling the LLM, so the test runs offline.
- However the test is marked `@pytest.mark.langsmith`, which is presumably a custom marker used to opt into traced runs against LangSmith (the marker is not registered in `pyproject.toml`).
- This test is **not** an end-to-end check of any real action path (`coaching`, `summary`, `qa`, etc.). It is a "graph compiles and invokes" smoke test inherited from the starter template.

### 8.3 Running tests locally

- `make test` → `python -m pytest tests/unit_tests/`.
- `make integration_tests` → `python -m pytest tests/integration_tests`.
- `make test_watch` → `python -m ptw --snapshot-update --now . -- -vv tests/unit_tests` (requires `pytest-watch`).
- `make extended_tests` → `python -m pytest --only-extended tests/unit_tests/` (requires the custom `--only-extended` plugin, not currently installed).

---

## 9. CI (GitHub Actions workflows)

Two workflows under `.github/workflows/`:

### 9.1 `unit-tests.yml` — name: **CI**

- **Triggers**: `push` to `main`, all `pull_request`s, and `workflow_dispatch`.
- **Concurrency**: cancels in-progress runs on the same `${{ github.workflow }}-${{ github.ref }}`.
- **Matrix**: `os = ubuntu-latest`, `python = [3.11, 3.12]`.
- **Steps**:
  1. `actions/checkout@v4`.
  2. `actions/setup-python@v4` with the matrix Python version.
  3. Install `uv` (curl-piped installer), create venv, `uv pip install -r pyproject.toml`.
  4. `uv pip install ruff && uv run ruff check .`.
  5. `uv pip install mypy && uv run mypy --strict src/`.
  6. `codespell-project/actions-codespell@v2` on `README.md`.
  7. `codespell-project/actions-codespell@v2` on `src/`.
  8. `uv pip install pytest && uv run pytest tests/unit_tests`.

### 9.2 `integration-tests.yml` — name: **Integration Tests**

- **Triggers**: scheduled (`cron: "37 14 * * *"`, i.e. 14:37 UTC daily ≈ 7:37 AM Pacific) and `workflow_dispatch`.
- **Concurrency**: same cancel-in-progress pattern.
- **Matrix**: `os = ubuntu-latest`, `python = [3.11, 3.12]`.
- **Steps**:
  1. Checkout, setup-python.
  2. Install `uv`, `uv venv`, `uv pip install -r pyproject.toml`, plus `pytest-asyncio`.
  3. Run `uv run pytest tests/integration_tests` with secrets piped through env:
     - `ANTHROPIC_API_KEY` — present for legacy reasons (the current code does not use Anthropic; see Section 12).
     - `LANGSMITH_API_KEY` — enables LangSmith tracing.
     - `LANGSMITH_TRACING=true`.

Importantly, **`OPENAI_API_KEY` is not injected** into the integration workflow. The current single integration test does not need it because the empty-transcript path short-circuits the LLM call. Any future test that exercises an LLM-bound action will fail in CI until this secret is added.

The README badges still point at the upstream `langchain-ai/new-langgraph-project` repository; they are not actually wired to this fork's workflows.

---

## 10. Deployment / Run

### 10.1 Local development — `langgraph dev`

Documented in `README.md`:

```bash
pip install -e . "langgraph-cli[inmem]"
cp .env.example .env   # add OPENAI_API_KEY, optional LANGSMITH_* keys
langgraph dev
```

`langgraph dev`:
- Reads `langgraph.json`, installs the `.` dependency, imports `./src/agent/graph.py:graph`.
- Boots a local LangGraph Server on `http://127.0.0.1:2024` (default) with an in-memory checkpointer (because we installed the `[inmem]` extra).
- Auto-creates a default Assistant for the `agent` graph.
- Hot-reloads on source changes.
- Exposes the LangGraph Studio UI URL on stdout for visual debugging.

`Makefile` does not provide a `make dev` shortcut; lint/test/format are the only convenience targets (Section 8.3, plus `make lint` / `make format` / `make spell_check`).

### 10.2 LangGraph Server / Studio

The compiled graph is registered with `name="Contact Center Agent"` (visible in Studio). Studio supports:
- Editing state and replaying from arbitrary checkpoints.
- Visualizing the conditional routing topology described in Section 4.5.
- Inspecting per-node inputs/outputs and LLM traces (if LangSmith is on).

For platform / cloud deployment, the `image_distro: "wolfi"` and `dependencies: ["."]` keys in `langgraph.json` are sufficient for the LangGraph Cloud builder (which produces a containerized server). No `Dockerfile` is checked in; image construction is handled by `langgraph build` / LangGraph Cloud.

### 10.3 HTTP API surface (LangGraph Server)

Once `langgraph dev` (or a deployed LangGraph Server) is running, the standard LangGraph Server REST surface is exposed at the server base URL. The endpoints most relevant to this graph are:

- **Assistants**
  - `POST /assistants` — create a new assistant bound to a graph id (here, `"agent"`). May embed `context` defaults.
  - `GET /assistants/{assistant_id}` — fetch.
  - `POST /assistants/search` — list / filter.

- **Threads**
  - `POST /threads` — create a stateful conversation thread (used by Studio's "follow-up extends the thread" flow).
  - `GET /threads/{thread_id}/state` — read latest state.
  - `POST /threads/{thread_id}/state` — patch state.
  - `GET /threads/{thread_id}/history` — list checkpoints.

- **Runs** (the most common entry point for backend services)
  - `POST /threads/{thread_id}/runs` — start an async run on an existing thread.
  - `POST /threads/{thread_id}/runs/wait` — start a run and block until completion, returning the final state.
  - `POST /threads/{thread_id}/runs/stream` — server-sent events stream of state/messages/events.
  - `POST /runs/stream` and `POST /runs/wait` — stateless (no thread) variants.

A typical invocation payload from a backend service looks like:

```http
POST /threads/{thread_id}/runs/wait
Content-Type: application/json

{
  "assistant_id": "agent",
  "input": {
    "action": "summary",
    "call_id": "call_123",
    "tenant_id": "tenant_abc",
    "agent_id": "agent_42",
    "transcript": "...",
    "duration_seconds": 312,
    "disposition": "resolved"
  },
  "context": {
    "tenant_id": "tenant_abc",
    "call_id": "call_123",
    "agent_id": "agent_42",
    "action": "summary"
  }
}
```

The final `state` returned by `/runs/wait` will contain the original input fields plus whichever of `sentiment`, `coaching_tip`, `summary`, `qa_score`, `lead_score`, `routing` were populated by the matched action path.

---

## 11. Internal Dependencies & Consumers

This repository has **no internal Python dependencies** on sibling projects. Its consumers, however, are external.

### 11.1 Parent monorepo

The Chappy Connect `telephony-service-bundle` aggregates four submodules (per `.gitmodules`):

| Submodule path | Upstream repo |
| --- | --- |
| `telephony-service/` | `ChappyAI/con-nexus-telephony` (NestJS backend). |
| `telephony-service-frontend/` | `ChappyAI/telephony-service-frontend`. |
| `langgraph-app/` | `ChappyAI/langgraph-contact-center` (**this repo**). |
| `crewai-contact-center/` | `ChappyAI/crewai-contact-center` (sibling CrewAI agent). |

The bundle's `AGENTS.md` and `CLAUDE.md` describe the relationship:

- `langgraph-app/` is mounted as a sibling Python project, run locally via `cd langgraph-app && pip install -e . "langgraph-cli[inmem]" && langgraph dev`.
- Inside the NestJS backend, the AI agency feature set is module-loaded as `AiAgencyModule` and gated by `FEATURE_AI_AGENCY` (default on) and `FEATURE_EXTERNAL_AGENCY` (default off — toggles CrewAI/LangGraph external connectors).
- The backend reads `LANGGRAPH_API_URL` (and a parallel `CREWAI_API_URL`) to locate this graph's running server. From bundle docs: "LangGraph/CrewAI connectors need `LANGGRAPH_API_URL`/`CREWAI_API_URL` env vars".

### 11.2 Expected invocation pattern

Although the NestJS source is not checked out in the working tree (the `telephony-service` directory is an empty submodule mount), the documented surface implies the standard LangGraph Server client flow:

1. NestJS service (likely under an `AiFeaturesModule` / `AiAgencyModule`) loads the LangGraph SDK or makes raw HTTP calls.
2. Per-call, it issues `POST /threads` (optional, for stateful conversations) then `POST /runs/wait` (or `/runs/stream` for live coaching SSE) against `LANGGRAPH_API_URL`, passing `assistant_id="agent"` and the action-specific payload outlined in Section 10.3.
3. The returned state's relevant subfield (`sentiment` / `summary` / `qa_score` / `lead_score` / `routing` / `coaching_tip`) is mapped onto the backend's domain models and surfaced through the AI-features REST endpoints described in the bundle's `CLAUDE.md`.

Realtime coaching is gated separately by `FEATURE_REALTIME_COACHING` on the NestJS side; when enabled, it presumably consumes the `/runs/stream` variant or repeatedly invokes the `coaching` action.

### 11.3 Other consumers

No other repository in the bundle currently invokes this graph directly. The CrewAI sibling (`crewai-contact-center`) is an independent alternative agent backend behind the same feature flag (`FEATURE_EXTERNAL_AGENCY`).

---

## 12. Known Gaps / Extension Points

The codebase carries some starter-template residue and a few minor inconsistencies that are worth flagging for future maintainers:

1. **Starter-template metadata not updated.**
   - `pyproject.toml` `name = "agent"`, version `0.0.1`, author still set to the original LangGraph template author.
   - `README.md` is the unmodified `new-langgraph-project` template README (CI badges point to `langchain-ai/new-langgraph-project`).
   - `INDEX.md` references an old local path (`/Users/seanchapman/DDev/...`) and an older `langgraph-app` name.
2. **Unused state fields.** `agent_id`, `caller_number`, and `metadata` are accepted into `State` but never read by any node. They are ready for downstream features (e.g. agent-aware coaching, caller-history lookups).
3. **`Runtime[Context]` is wired but unread.** No node reads `runtime.context`. Action selection currently relies on `state.action` only. If clients pass `Context.action` rather than `State.action`, the graph will silently default to `sentiment`.
4. **`route_by_action` sends `summary` through `analyze_sentiment`.** This costs an extra LLM call and may not be intentional — the summary prompt already asks for `sentiment_overall`.
5. **Empty-transcript branch in `analyze_sentiment`** returns `{"sentiment": "neutral", "score": 0.5, "emotions": []}` with no `trend` / `alert` keys. Downstream `generate_coaching` uses `.get(...)` so it tolerates this, but external consumers reading `state.sentiment` should not assume those fields are always present.
6. **`score_quality.overall_score` scale.** Sub-scores are 0–10; `overall_score = int((g+e+r)/3 * 10)` therefore lives on a 0–100 scale (max ≈ 83 for a perfect call). Document the scale before exposing it to UI.
7. **No structured-output binding.** Sentiment and summary nodes rely on prompt-only JSON contracts plus `json.loads` with broad fallbacks. Migrating to `llm.with_structured_output(<pydantic-model>)` or response-format JSON mode would harden these paths.
8. **No retries / no error surfacing.** `State.error` is declared but never written. LLM failures (network, rate limit) propagate as exceptions. There is no per-node retry or graceful fallback into `state.error`.
9. **Test coverage is essentially zero for business logic.** The only assertions are "graph imports" and "ainvoke returns non-None on an empty payload". Add per-node unit tests (rule-based nodes are easy targets) and an action-by-action integration matrix.
10. **CI vs. runtime mismatch.** `integration-tests.yml` injects `ANTHROPIC_API_KEY` (unused) but not `OPENAI_API_KEY` (required for any LLM-bound test). Update the secret set before adding meaningful integration tests.
11. **Optional dependencies divergence.** Both `[project.optional-dependencies].dev` and `[dependency-groups].dev` exist in `pyproject.toml`. The latter is PEP 735 dependency groups (used by `uv`); the former is legacy. Consolidate to avoid drift.
12. **README badges and instructions** describe the starter, not the contact-center semantics. Replacing the README with action/route/state documentation (this file is a starting point) would help downstream integrators.
13. **No graph-level checkpointer / store config**, which is correct for delegation to the LangGraph server. If self-hosting outside the server, the consumer must compile with an explicit `checkpointer=` to retain state across runs.

---

## 13. File Index

Repository-relative paths (resolved against the repo root wherever it is checked out):

| Path | Role |
| --- | --- |
| `src/agent/graph.py` | Sole graph implementation: state, nodes, routing, compiled `graph` symbol. |
| `src/agent/__init__.py` | Re-exports `graph` so `from agent import graph` works. |
| `langgraph.json` | LangGraph Server config; maps `agent` -> `./src/agent/graph.py:graph`, sets env file and image distro. |
| `pyproject.toml` | Project metadata, runtime + dev deps, ruff + mypy config, setuptools package mapping. |
| `Makefile` | `test`, `integration_tests`, `test_watch`, `lint`, `format`, `spell_check` targets. |
| `.env.example` | Template for `.env` (LANGSMITH_PROJECT placeholder). |
| `.gitignore` | Ignores `.env`, `.langgraph_api/`, IDE/AI tool caches, build artifacts. |
| `.codespellignore` | Empty codespell ignore list. |
| `LICENSE` | MIT license. |
| `INDEX.md` | Legacy starter-template orientation doc. |
| `README.md` | Original starter README (not yet rewritten for contact-center). |
| `uv.lock` | uv-resolved dependency lockfile. |
| `.github/workflows/unit-tests.yml` | Push/PR CI: ruff, mypy --strict, codespell, pytest unit tests, Python 3.11/3.12 matrix. |
| `.github/workflows/integration-tests.yml` | Daily cron + manual integration tests; injects LangSmith + (legacy) Anthropic secrets. |
| `tests/conftest.py` | Session-scoped `anyio_backend = "asyncio"` fixture. |
| `tests/unit_tests/__init__.py` | Package marker. |
| `tests/unit_tests/test_configuration.py` | Asserts compiled `graph` is a `langgraph.pregel.Pregel`. |
| `tests/integration_tests/__init__.py` | Package marker. |
| `tests/integration_tests/test_graph.py` | Smoke test: `await graph.ainvoke({"changeme": "some_val"})`. |
| `static/studio_ui.png` | README screenshot of LangGraph Studio. |

---

*End of reference.*
