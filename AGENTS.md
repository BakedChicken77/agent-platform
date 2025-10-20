# Agent Platform

**Agent Platform** is a full toolkit for running AI agent workflows built with **LangGraph**, **FastAPI**, and **Streamlit**. It provides an end-to-end system to define AI agents, serve them via a web API, and interact through a web UI or client library. Key components include:

- **LangGraph Agents** – AI agents defined as stateful workflows (graphs) using the LangGraph framework.
- **FastAPI Service** – A web service that hosts agents with both streaming and non-streaming endpoints.
- **Streamlit UI** – A chat-based user interface for interacting with the agents in real time.
- **Agent Client** – A Python client library to programmatically invoke agents.
- **Pydantic Schema** – Data models for requests/responses and configuration, ensuring type-safe interaction.

This project allows you to easily build and run custom AI agents by extending the provided framework. Multiple agents can run concurrently, and each agent can maintain short-term conversation state and long-term memory.

## Build & Commands

Use the following commands to set up and run the project in a development environment:

To install all dependencies:
```sh
uv sync --group dev --group client
```

Use `uv run` or first activate `.venv` with `source .venv/bin/activate`

To run frontend:
```sh
uv run streamlit run src/streamlit_app.py
```

To run backend:
```sh
uv run fastapi dev src/service/service.py
```

Always run the regression suite before submitting a change:
```bash
uv run pytest
```


**Common Commands:**

- `uv sync --frozen` – Install/update project dependencies (pinning versions from `uv.lock`).
- `pytest` – Run the test suite (see **Testing** below).
- `pre-commit run --all-files` – Run linters/formatters on the codebase (ensures code style compliance).

## Code Style

This project follows standard Python coding conventions (PEP 8) with additional rules enforced via **Ruff** and **mypy**:

- **Type Checking:** The code is fully type-hinted and checked with `mypy` for consistency. Use type hints in all function signatures and dataclass/model fields.
- **Linting:** Ruff is used for linting (configured for a **100 character** line length). It also covers most formatting concerns. Run `ruff --fix .` to autofix simple issues.
- **Formatting:** Consistent code style is enforced. Use 4 spaces for indentation, and single quotes for strings where possible. The repository may include a `pre-commit` hook (installed via `pre-commit install`) to run formatting and lint checks on commit.
- **Naming:** Follow Python naming conventions (snake_case for functions/variables, PascalCase for classes). Use descriptive names for agents, functions, and variables to clarify their purpose.
- **Imports:** Organize imports into standard library, third-party, and local sections. Avoid unused imports (Ruff will flag them).

In general, strive for readable, self-documenting code. Comments or docstrings should be added for complex logic or non-obvious behavior, especially in agent definitions and core utilities.

## Testing

The project includes a comprehensive test suite (unit and integration tests) to ensure all components work together:

- **Test Framework:** Uses **pytest** (with plugins `pytest-asyncio` for async code and `pytest-cov` for coverage). Tests are located in the `tests/` directory.
- **Running Tests:** Activate the virtual environment and simply run `pytest`. By default, tests will use a dummy OpenAI API key (set via `PYTEST_ENV` in `pyproject.toml`) and an in-memory SQLite database for agent state. All tests should pass without external dependencies if configured correctly.
- **Test Structure:** 
  - Unit tests cover individual modules (e.g., testing schema models serialization, tools behavior, client methods).
  - Integration tests cover higher-level scenarios, such as API endpoint responses and the Streamlit app’s functionality (using FastAPI’s TestClient or Streamlit’s testing utilities).
- **Writing Tests:** When adding new features or fixing bugs, include new tests. Tests should be deterministic and not rely on actual external API calls (use fake or local model providers where possible).

## Security

Security and responsible usage considerations for this agent platform:

- **API Keys & Secrets:** *Never commit API keys or secrets to the repository.* All sensitive credentials (LLM API keys, database passwords, etc.) must be provided via environment variables in the `.env` file. The project uses **python-dotenv** to load these at runtime.
- **Authentication:** The FastAPI service supports JWT-based authentication via Azure AD. By default, `AUTH_ENABLED=True`, meaning each request is expected to include a valid Bearer token. In development (and in Docker by default), authentication can be disabled (`AUTH_ENABLED=False`) for convenience. When enabled, the service verifies tokens against Azure AD and rejects unauthorized requests (see `src/auth/`).
- **CORS:** Cross-origin requests are allowed (`allow_origins=["*"]` by default) for development flexibility. In production, consider restricting `origins` or implementing stricter CORS policies to only allow trusted domains.
- **Content Moderation:** The platform can integrate with **LlamaGuard** for AI output safety. If a Groq API key is provided (`GROQ_API_KEY`), certain agents will automatically filter content. For example, the `excel_agent` uses `LlamaGuard` to intercept unsafe outputs. This helps ensure the AI does not produce disallowed content. Always review and adjust safety filters as appropriate for your use case.
- **File Uploads:** The service allows users to upload files for agents to use (via the `/files/upload` endpoint). Uploaded files are stored under an `uploads/` directory (or in a database for production) segregated by user. Only certain file types are permitted (see `ALLOWED_MIME_TYPES` in settings). The upload size is capped (`MAX_UPLOAD_MB`, default 100 MB per file). These measures help prevent abuse (e.g., disallowing executables). Always validate and sanitize any file inputs on the agent side.
- **Database Security:** If using Postgres or MongoDB for agent memory, ensure the database is secured (use strong credentials, network restrictions). The platform defaults to SQLite for simplicity, but production deployments should prefer a robust external database with proper access controls.
- **Production Deployment:** When deploying the service, run behind HTTPS (FastAPI can be behind a proxy like Nginx/Traefik). Set `MODE=production` (if used) and ensure that debug or test flags are off. Regularly update dependencies to include security patches.

## Architecture & Design

**Project Architecture:** This repository is organized into logical components for clarity and modularity:

- **`src/agents/`** – **Agent Definitions.** Each agent is a LangGraph workflow (state machine) that defines how the AI should respond to inputs. For example, `chatbot.py` defines a simple conversational agent, while `universal_supervisor_agent.py` defines a multi-expert supervisor agent. Agents can use Tools and may have memory (see below).
- **`src/service/`** – **FastAPI Service.** Contains the FastAPI app (`service.py`) which exposes HTTP endpoints to interact with agents. It sets up routes for invocation (`POST /{agent}/invoke`), streaming (`POST /{agent}/stream`), retrieving info (`GET /info`), file uploads (`POST /files/upload`), etc. The service also initializes persistent **memory** for agents at startup (using either SQLite, Postgres, or MongoDB, depending on configuration). It attaches a **checkpointer** (short-term conversation memory) and a **store** (long-term memory/knowledge base) to each agent instance.
- **`src/core/`** – **Core Utilities and Configuration.** This includes configuration management (`settings.py` with Pydantic for environment variables), LLM model loading (`llm.py` selects the appropriate model class for a given model name), and middleware (e.g., `LoggingMiddleware` for HTTP logging). It also contains integration logic for telemetry (e.g., `core/langgraph.py` to instrument agent execution with Langfuse for tracing).
- **`src/schema/`** – **Data Models.** Pydantic models defining the schema for requests and responses. This includes classes like `UserInput`, `ChatMessage`, `AgentInfo`, `ServiceMetadata`, etc., which ensure that data passing in and out of the service is well-defined. These models are used both by the FastAPI endpoints (for validation and documentation) and by the AgentClient.
- **`src/client/`** – **Client Library.** A Python module (`client.py`) providing the `AgentClient` class. This client wraps HTTP calls to the FastAPI service, offering convenient methods to invoke agents synchronously or asynchronously, stream responses, and retrieve available agents. Developers can use this in their own Python applications to integrate with the agent service.
- **`src/auth/`** – **Authentication Layer.** Contains middleware and utilities for Azure AD OAuth2 JWT validation. The `AuthMiddleware` ensures that incoming requests have a valid token (unless auth is disabled or the route is whitelisted). This module is crucial for securing the service in a multi-user or production environment.
- **`src/tools/`** – **Tool Implementations.** (If present) Utility modules that implement Tools which agents can use. For example, you may find tools for interacting with Excel files, performing web searches, calculator functions, etc. Agents bind to these tools to extend their capabilities (e.g., an agent can call `write_to_excel` to output data to a spreadsheet).

**Agent Memory & State:** Each agent can maintain conversational state. On each request, the service supplies a **thread_id** and **user_id** to the agent’s runtime configuration. Short-term state (conversation history) is checkpointed via a **Checkpointer** (in-memory or database-backed) so the agent can remember past messages in a session. Long-term state (knowledge or context across sessions) can be stored via a **Store** (e.g., vector database for embeddings) if configured. By default, `DATABASE_TYPE` is "sqlite", so agent state is kept in `checkpoints.db` locally. For production or multi-instance scenarios, switch to Postgres or Mongo for state persistence.

**Multiple Agents:** The platform supports running multiple agents simultaneously. The FastAPI routes are parameterized by agent name – for example, `POST /chatbot/invoke` vs `POST /coding_expert/invoke`. The `/info` endpoint returns all available agent names and their descriptions so clients know what agents are offered. Internally, agents are registered in a dictionary (`agents` in `agent_registry.py`) at startup. Adding a new agent is straightforward (see **src/agents** documentation).

**Streaming Responses:** The service supports real-time streaming of agent responses. When a client calls `POST /{agent}/stream`, the response is sent as a stream of events. This allows token-by-token streaming for model outputs and intermediate tool messages. The implementation uses server-sent events (SSE) with data chunks prefixed by `data:` lines. The client library handles these streaming responses and reconstructs messages. Streaming is useful for a responsive UI, as the user can see partial answers while the agent is working.

**Extensibility:** The Agent Platform is designed to be extensible:
- You can define new agents (with specialized capabilities or tools) and register them to expose new endpoints.
- You can add new Tools for agents to use (e.g., integration with external APIs or databases) by implementing functions and decorating with `@tool`.
- The system supports various LLM backends (OpenAI, Azure OpenAI, Anthropic, Ollama, etc.) through LangChain/LangGraph integrations. By adjusting environment variables, you can switch the model or provider without code changes.
- Tracing and logging are integrated (LangSmith, Langfuse). You can tie in other observability tools if needed by building on the patterns in `core/langgraph.py`.

## Configuration

Configuration is managed via environment variables, loaded into `core.settings.Settings` (a Pydantic settings class). Key configuration options include:

- **LLM Provider API Keys:** At least one of `OPENAI_API_KEY`, `GROQ_API_KEY`, `OLLAMA_MODEL` (for local Ollama), `AZURE_OPENAI_API_KEY` (with `AZURE_OPENAI_ENDPOINT`), etc., must be set. If no model credentials are provided, the service will refuse to start. The first available provider will determine the `DEFAULT_MODEL` and available models list.
- **Model and Provider Selection:** You can explicitly set `DEFAULT_MODEL` to a value from the enumerated model names (see `schema.models.AllModelEnum`). Otherwise, the system picks a default (e.g., GPT-4 or a mini model for OpenAI, LLaMA 2 for Groq, etc.). To use Azure OpenAI, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and optionally map model names via `AZURE_OPENAI_DEPLOYMENT_MAP`. For Ollama, set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` to use a local model.
- **Database Settings:** `DATABASE_TYPE` controls where agent state is stored. Options: `"sqlite"` (default, local file `checkpoints.db`), `"postgres"`, or `"mongo"`. For Postgres, set `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, and importantly `PGVECTOR_URL` (a connection string including the `pgvector` extension, used for vector store if needed). For Mongo, set `MONGO_HOST`, `MONGO_PORT`, etc. (Note: at this time, long-term memory store for Mongo may not be fully implemented, and Postgres is recommended for persistent memory).
- **Server Settings:** `HOST` (default `http://localhost`) and `PORT` (default `8080`) control the FastAPI server host. These are used in development when running `run_service.py`. In Docker, these are set in compose and typically remain default. **Do not expose the dev server publicly without proper auth and HTTPS.**
- **Authentication:** `AUTH_ENABLED` (boolean, default `True`). If True, the Auth middleware requires OAuth2 JWTs on each request. Azure AD settings: `AZURE_AD_TENANT_ID`, `AZURE_AD_CLIENT_ID` (application’s client ID for audience validation), and `AZURE_AD_API_CLIENT_ID` (for constructing the expected `api://` audience). The system will fetch the JWKS keys from Azure based on the tenant ID. To disable auth (for local testing or if you wrap the service with your own auth proxy), set `AUTH_ENABLED=False`.
- **Feature Toggles:** 
  - `LANGCHAIN_TRACING_V2` and `LANGCHAIN_ENDPOINT`, `LANGCHAIN_API_KEY`: enable LangSmith tracing of agent runs.
  - `LANGFUSE_TRACING` and `LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY`: enable Langfuse tracing for agent steps (the code is prepared to send spans to Langfuse if configured).
  - `USE_FAKE_MODEL`: if true, allows using a fake LLM for testing (responds with preset messages).
  - `OPENWEATHERMAP_API_KEY`: if set, the example weather tool in some agents (e.g., Excel agent) can fetch real weather data.
- **Miscellaneous:** `WHITELIST` can list specific routes to exclude from auth (e.g., health checks or a playground route). `MAX_UPLOAD_MB` (default 100) sets the upload size limit for file uploads. Logging levels can be adjusted via the `LOGLEVEL` environment variable (if implemented in settings).

For a full list of configuration variables, see the provided `.env.example` file and `src/core/settings.py`. After changing environment variables, you will need to restart the service to apply them.

**Note:** When adding new configuration options, update the `.env.example` file and the `Settings` class, and document their purpose. Consistent and well-documented configuration helps both developers and AI agents understand how to run and modify the system.

