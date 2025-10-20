# Core Module (src/core)

The core module contains foundational utilities and configuration for the Agent Platform. It is responsible for managing application settings, initializing language model interfaces, and providing middleware or hooks that other parts of the system use. Essentially, **src/core** is the glue that ties together environment configuration, logging, and external integrations.

## Purpose

The core module’s purpose is to provide a central place for:
- **Configuration Management:** Loading and validating environment variables, and making configuration accessible throughout the app.
- **Model Management:** Creating or retrieving instances of language model clients (LLMs) based on configured providers.
- **Middleware and Logging:** Implementing any cross-cutting concerns (e.g., HTTP logging, request handling enhancements).
- **Integration Hooks:** Code that integrates external services or monitoring.
- **Shared Utilities:** Any utility functions or classes used by multiple agents or services (if not large enough to need their own module).

By isolating these concerns in core, the rest of the code (agents, service, etc.) can remain focused on their domain logic and simply import what they need from core.

## Configuration (Settings.py)

The `core/settings.py` file defines the **Settings** class using Pydantic’s BaseSettings. This class declares all the environment variables and their default values or types:
- Basic server config: `HOST`, `PORT`, `MODE`.
- Authentication config: `AUTH_ENABLED` and Azure AD details (`AZURE_AD_TENANT_ID`, `AZURE_AD_CLIENT_ID`, etc.).
- API keys for various providers: `OPENAI_API_KEY`, `GROQ_API_KEY`, `OLLAMA_BASE_URL`/`OLLAMA_MODEL`, flags like `USE_FAKE_MODEL`.
- Default and available model enums: `DEFAULT_MODEL` (of type `AllModelEnum`), and `AVAILABLE_MODELS` (populated at runtime in `model_post_init`).
- Optional compatible API (if using a custom model endpoint): `COMPATIBLE_MODEL`, `COMPATIBLE_API_KEY`, `COMPATIBLE_BASE_URL` (this was for OpenAI-compatible proxies).
- External service keys: `OPENWEATHERMAP_API_KEY`, LangSmith and LangFuse toggles and keys (`LANGCHAIN_TRACING_V2`, `LANGFUSE_TRACING`, etc.).
- **Database settings:** `DATABASE_TYPE` (an enum for "sqlite", "postgres", "mongo"), along with connection details for each (SQLite file path, Postgres user/password/host/port/db, Mongo URI parts). Also `PGVECTOR_URL` which is a convenient single connection string for Postgres with pgvector.
- **Uploads config:** `UPLOAD_DIR` (default `"./uploads"`), `MAX_UPLOAD_MB` (max file upload size), and `ALLOWED_MIME_TYPES` (the set of MIME types allowed for file upload).
- Many of these settings have defaults suitable for development (e.g., SQLite and `AUTH_ENABLED=True`, which still works with no tokens if not enforced manually).
- Pydantic is configured to load from `.env` (with `env_file = find_dotenv()` so it finds the project’s .env file), and to ignore extra env vars.

After environment variables are loaded, the Settings class performs a **post-init** (`model_post_init` method). This method:
- Checks that at least one LLM API key or provider is configured (it collects all providers and if none is present, it raises an error and prevents the app from starting).
- It then populates the `DEFAULT_MODEL` if not set: depending on which provider’s key is present, it chooses a default model (e.g., if OpenAI key is set, default to a specific OpenAI model like GPT-4O; if only Groq is set, default to a Groq model, etc.).
- It also fills the `AVAILABLE_MODELS` set with all model enums from the active providers (so if OpenAI and Groq are both configured, it includes all OpenAI and Groq model names in the allowed list). This list is returned in the `/info` endpoint and used by agents to validate model choices.

The Settings object is used by importing `core.settings` (which instantiates a singleton `settings = Settings()` at the bottom of the file). Anywhere in the code, `from core import settings` gives access to all config values (as attributes, e.g., `settings.OPENAI_API_KEY`).

This design centralizes configuration. If you add a new environment variable, add it to Settings and it becomes globally available.

## Model Loader (LLM.py)

The `core/llm.py` module defines how to get a language model instance given a model name. It uses LangChain/LangGraph model classes:
- It imports various Chat model classes: `ChatOpenAI` (OpenAI API), `AzureChatOpenAI`, `ChatGroq` (for Groq’s LLaMA model), `ChatOllama` (for local Ollama models), etc. Additionally, it defines a `FakeToolModel` (subclassing LangChain’s FakeListChatModel) for testing purposes.
- `_MODEL_TABLE`: A dict mapping each model enum (from `schema.models`) to a provider-specific string (like model IDs). This is built by combining multiple enumerations (OpenAI, Azure, Groq, Ollama, etc.).
- The main function: `get_model(model_name: AllModelEnum) -> ModelT`:
  - It looks up the `api_model_name` via `_MODEL_TABLE` for the given enum. If not found, it raises an error (unsupported model).
  - Depending on which set the model belongs to, it initializes the appropriate class:
    - If `model_name` is an OpenAI model (e.g., GPT-4), it returns `ChatOpenAI(model=<name>, temperature=settings.TEMPERATURE, streaming=True)`.
    - If Azure OpenAI, it requires that `AZURE_OPENAI_API_KEY` and `ENDPOINT` are set, then returns `AzureChatOpenAI( ... )` with the deployment name and API version.
    - If Groq, uses `ChatGroq`. For a special case `LLAMA_GUARD_4_12B` model, it sets temperature 0 (since that model is specifically for content moderation).
    - If Ollama, it constructs `ChatOllama` – if `OLLAMA_BASE_URL` is provided, it uses it; otherwise it assumes a local default. It uses `settings.OLLAMA_MODEL` as the model name to load.
    - If Fake model, it returns an instance of `FakeToolModel` which will produce a fixed dummy response.
    - (Other providers like Anthropic, VertexAI, etc., are stubbed out in comments – showing the intent to add them later by similar patterns.)
  - The returned object is a LangChain `ChatModel` which the agents will call (via `model.ainvoke()` typically).
- This factory function is **cached** with `functools.cache`, meaning repeated calls for the same model_name will reuse the same model instance (where applicable). This is good for performance (avoids re-authenticating or loading heavy models repeatedly).
- The use of `settings.TEMPERATURE` means you can globally control the creativity of all models by tweaking that in .env.

By using `get_model`, agents do not need to worry about API details. They just call `model = get_model(model_name)` to get a ready-to-use model. This also encapsulates the logic of which model streams tokens (in our setup, all returned models have streaming=True support if applicable).

If new model providers need to be supported, this is the place to update:
- Add the new provider’s model class import,
- Extend `schema.models` with its enum and values,
- Update `_MODEL_TABLE` to include those enums,
- Extend `get_model` with a case for that provider.

## LoggingMiddleware

The `core/logging_middleware.py` defines `LoggingMiddleware`, a Starlette BaseHTTPMiddleware that logs each request:
- On each request, it notes the start time, then awaits `call_next(request)` to get the response, then calculates latency.
- It logs a structured message with method, path, status code, latency, client IP, and user agent.
- The logger is named `"http"` and can be configured in the logging config to output to console or file as needed. By default, FastAPI uses `logging` module; our usage here means these logs will appear if logging is set to INFO level.
- We add this middleware to the FastAPI app in `service.py`, so every API request is captured. This is useful for debugging and monitoring usage patterns.



Thus, `core/__init__.py` imports `settings = Settings()`, `get_model` from llm, and `LoggingMiddleware` class, making them directly accessible.

The **dependencies** of core are notable:
- **pydantic and pydantic_settings** for config.
- **langchain_core and related** for model classes.
- Possibly **requests** if Langfuse or others require HTTP calls (Auth middleware uses requests to fetch OIDC config in `auth`, which is separate).
- Standard libraries: json, os, logging, etc.

## How Core Contributes

The core module is not run by itself; instead, it’s imported and used by:
- **Service**: to get config (`settings`), to attach middleware (`LoggingMiddleware`).
- **Agents**: to get an LLM (`get_model`) and to instrument their nodes. Also, agents use `settings` for global parameters like `settings.TEMPERATURE` or to check feature flags.
- **Auth**: technically part of core concerns, but in our repo it’s split into `src/auth`. Still, core provides the data (JWKS URL, etc.) via settings to the auth component.

Whenever you adjust something in core (like adding a new environment variable or changing the model loading logic), ensure that:
- The change is compatible with existing usage (e.g., if you change default `TEMPERATURE`, know that it affects all agents uniformly).
- Update documentation (this file or root AGENT.md) if it’s a significant change.
- Reflect the new variable in `.env.example` with a description.

## Examples

- **Using get_model:** In an interactive Python session (with .env loaded), you can do:
  ```python
  from core.llm import get_model
  from schema.models import OpenAIModelName
  model = get_model(OpenAIModelName.GPT_4)  # requires OPENAI_API_KEY in env
  print(model)  # e.g., ChatOpenAI instance ready to use
  response = model("Hello")  # make sure to call in an async loop if using .ainvoke
