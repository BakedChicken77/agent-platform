# Contributing

Thanks for taking the time to improve the Agent Platform! This guide summarises the
minimum steps for getting a local environment running and explains how to keep the
Langfuse instrumentation healthy while you work.

## Local development checklist

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Copy the environment template**
   ```bash
   cp .env.example .env
   ```

3. **Populate required secrets**
   At minimum you need an LLM key (for example `OPENAI_API_KEY`). To exercise the
   Langfuse instrumentation locally, provide the tracing keys as well:

   ```env
   LANGFUSE_TRACING=true
   LANGFUSE_HOST=https://cloud.langfuse.com
   LANGFUSE_PUBLIC_KEY=pk-your-project
   LANGFUSE_SECRET_KEY=sk-your-project
   LANGFUSE_CLIENT_PUBLIC_KEY=pk-your-client  # optional, enables Streamlit trace links
   LANGFUSE_ENVIRONMENT=development           # optional label used in dashboards
   ```

4. **Run the services**
   ```bash
   uv run streamlit run src/streamlit_app.py
   uv run fastapi dev src/service/service.py
   ```

## Running tests and linters

Always run the regression suite before submitting a change:

```bash
uv run pytest
```

Using `uv run` (or first activating `.venv` with `source .venv/bin/activate`) is
important—invoking `pytest` with the system interpreter will skip the synced
dependencies and surface `ModuleNotFoundError` messages for packages such as
`langchain_core`, `pydantic`, `httpx`, `streamlit`, `fastapi`, or `starlette`.

The repository ships with Langfuse-aware tests that mock the SDK. No external
network calls are required.
