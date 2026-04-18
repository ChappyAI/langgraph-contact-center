# langgraph-app - Development Overview

## 1. Purpose and Scope
This project is a starter template for creating an AI agent state machine using LangGraph. It is designed to demonstrate how to get started with the LangGraph Server and LangGraph Studio, a visual IDE for debugging LangGraph applications. The core logic defines a simple, single-step application that can be extended to orchestrate more complex agentic workflows.

## 2. Key Technologies and Dependencies
- **Language/Runtime**: Python (>=3.10)
- **Framework**: LangGraph
- **Key Libraries**:
    - `langgraph`: The core framework for building stateful, multi-actor applications with LLMs.
    - `langchain-openai`: Integration with OpenAI's language models.
    - `python-dotenv`: For managing environment variables.
    - `langgraph-cli`: For running the LangGraph development server.
- **Internal Dependencies**: This is a self-contained project with no dependencies on the other sub-projects in the monorepo.

(Reference: [pyproject.toml](file:///Users/seanchapman/DDev/containers/telephony/luhx/langgraph-app/pyproject.toml))

## 3. Architecture and Core Components
- **Entry Points**: The main application logic is defined in `src/agent/graph.py`.
- **Core Components**:
    - `src/agent/graph.py`: Contains the definition of the LangGraph graph, including its nodes and edges. This is where the agent's workflow is orchestrated.
    - `langgraph.json`: Configuration file for the LangGraph server.

## 4. Setup and Local Development Environment
- **Prerequisites**: Python 3.10+.
- **Installation/Build**:
    ```bash
    pip install -e . "langgraph-cli[inmem]"
    ```
- **Running Locally**:
    ```bash
    langgraph dev
    ```
    This command starts the LangGraph server and provides access to the LangGraph Studio IDE.

## 5. Important Documentation Links
- [README.md](file:///Users/seanchapman/DDev/containers/telephony/luhx/langgraph-app/README.md): Provides instructions on how to get started with the project, including installation, running the server, and customizing the agent graph.

## 6. Testing Strategy
The project includes unit and integration tests located in the `tests/` directory.
- `tests/unit_tests/`: Contains unit tests for the application.
- `tests/integration_tests/`: Contains integration tests for the graph.
- Tests are run via GitHub Actions, as configured in `.github/workflows/`.

## 7. Potential Areas for Improvement / Future Work
This is a starter template, so future work would involve expanding the graph in `src/agent/graph.py` to create a more sophisticated agent. This could include adding more nodes for different tools or conditional edges for more complex decision-making.

## 8. Codebase Structure (Top 4 Levels)
```
.
├── .github/
│   └── workflows/
├── src/
│   └── agent/
├── static/
├── tests/
│   ├── integration_tests/
│   └── unit_tests/
├── .env.example
├── Makefile
├── README.md
├── langgraph.json
└── pyproject.toml
```
