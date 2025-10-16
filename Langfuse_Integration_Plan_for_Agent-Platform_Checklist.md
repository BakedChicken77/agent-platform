### ✅ **Step 1 Checklist Review**

#### 1. **Create `src/core/langfuse.py` helper**

**✔ Done.**

* Implemented a lazily cached `get_langfuse_client()`, `get_langfuse_handler()`, and `create_span()` factory.
* Handles optional Langfuse SDK gracefully (with local imports, error handling).
* Uses `settings.LANGFUSE_*` and guards against missing secrets.
* Uses `nullcontext` when tracing is disabled (ensures app start safety).

#### 2. **Extend `Settings` and `.env.example`**

**✔ Done.**

* `Settings` model updated with:

  * `LANGFUSE_TRACING`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_CLIENT_PUBLIC_KEY`, `LANGFUSE_ENVIRONMENT`, `LANGFUSE_SAMPLE_RATE`
* `.env.example` extended accordingly.
* Types and defaults handled correctly, e.g., `SecretStr` and nullable strings/floats.

#### 3. **Update `compose.yaml` + Dockerfiles**

**✔ Done.**

* `compose.yaml`:

  * Added `LANGFUSE_*` to both `build.args` and `environment` for `agent_service` and `streamlit_app`.
* `Dockerfile.app` and `Dockerfile.service`:

  * Accept all `LANGFUSE_*` as `ARG`s and promote to `ENV`.

#### 4. **Document in `README.md`**

**✔ Done.**

* Added a clear “Enabling Langfuse Tracing” section:

  * Includes setup steps, `.env` example, restart instructions, and `/health` check usage.

#### 5. **Guard application when tracing is off**

**✔ Done.**

* `get_langfuse_client()` returns `None` if tracing is disabled or SDK not installed.
* `create_span()` falls back to `nullcontext()`.
* `get_langfuse_handler()` returns `None` if SDK or keys are missing.
* `/health` route defends against failed SDK loading or client errors.

#### 6. **Basic regression test coverage**

**✔ Done.**

* Added `tests/core/test_langfuse.py` with 3 robust test cases:

  * No SDK / no tracing
  * Simulated SDK failure
  * Dummy SDK with env variable overrides
* Uses `monkeypatch`, `sys.modules`, and `pytest` fixtures properly.
* Clears internal caches to ensure clean test environments.

### ✅ **Step 2 Checklist Review**

#### 1. **Instrument `service.py` LangChain entrypoints**

**✔ Done.**

* Wrapped `agent.ainvoke()` inside a Langfuse span that records input/output metadata and tags `agent_id`, `thread_id`, and `user_id` via `LangfuseTelemetry`. Implemented in `service.py`'s `invoke` handler.

#### 2. **Attach Langfuse handler to `RunnableConfig`**

**✔ Done.**

* Added `_prepare_langfuse_context()` to build per-request telemetry, configure callback handlers with `trace_id`, `session_id`, and `user_id`, and inject them into `RunnableConfig.callbacks` inside `_handle_input()`.

#### 3. **Graceful exception handling**

**✔ Done.**

* Introduced `_end_span()` helper to flag errors with `level="ERROR"` and status messages, invoked from both `invoke` and streaming flows so traces remain inspectable after failures.

#### 4. **Stream tracing for `message_generator`**

**✔ Done.**

* `message_generator()` now establishes an `agent.stream` span and emits child spans per SSE message/token via `_record_stream_message_span()` / `_record_stream_token_span()` for deep streaming visibility.

#### 5. **Optimize `/health` Langfuse check**

**✔ Done.**

* Added `_get_cached_langfuse_health()` with a 30-second TTL cache so `/health` reuses recent Langfuse connectivity results before calling `auth_check()` again.

