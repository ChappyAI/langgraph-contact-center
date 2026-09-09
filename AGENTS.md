# AGENTS.md — langgraph-app

Canonical project instructions live in **`CLAUDE.md`** (same directory) — read it before working here. Essentials:

- **LangGraph state machine** (`src/agent/graph.py` — single file defines the whole graph) serving 6 contact-center call actions for `telephony-service` via `LANGGRAPH_API_URL`. Part of the `luhx/` workspace; this dir is a **git submodule**.
- **Package manager is `uv`** (never bare pip): `uv sync`, `uv run pytest`, `uv run langgraph dev`. `unset VIRTUAL_ENV` if a stale parent venv is inherited.
- Lint gate: `uv run ruff check . && uv run mypy --strict src/`. Integration tests hit OpenAI — need `OPENAI_API_KEY` (pytest does NOT auto-load `.env`).
- LLM nodes parse JSON with safe-default fallbacks — never tighten to hard failures (runs against live calls). Coaching/summary/lead-score intentionally route through sentiment first; don't shortcut.
- **Deploy**: Fly app `chappy-lang-api` (:8002, `/health`), `fly deploy --remote-only`; auto-deployed on super-repo `main` push when changed. Local: PM2 process `chappy-lang-api` (:8002).
- No AI attribution in commits; Conventional Commits.
