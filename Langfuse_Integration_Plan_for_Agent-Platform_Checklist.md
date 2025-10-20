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

### ✅ **Step 3 Checklist Review**

#### 1. **Inject trace/session context into LangGraph agents**

**✔ Done.**

* `_handle_input()` now adds `agent_name`, `run_id`, and a consolidated `langfuse` payload into `RunnableConfig.configurable`, ensuring every graph receives `trace_id`, `session_id`, and identity metadata. Implemented in `src/service/service.py`.

#### 2. **Decorate nodes with Langfuse spans**

**✔ Done.**

* Introduced `core.langgraph.instrument_langgraph_node()` to wrap LangGraph callables/runnables with Langfuse node spans, and applied it to chatbot, coding, supervisor, and background task nodes.

#### 3. **Persist Langfuse IDs across checkpoints**

**✔ Done.**

* Added `persist_langfuse_state()` / `ensure_langfuse_state()` helpers that stash `trace_id`, `run_id`, and `session_id` inside graph state/checkpoints (`src/agents/chatbot.py`, `src/agents/coding_agent.py`, `src/agents/bg_task_agent/bg_task_agent.py`).

#### 4. **Regression coverage for LangGraph helpers**

**✔ Done.**

* Added `tests/core/test_langgraph.py` verifying span instrumentation metadata and state persistence merging.

### ✅ **Step 4 Checklist Review**

#### 1. **Instrument LangChain tools with Langfuse telemetry**

**✔ Done.**

* Added `core/tools.py` with `instrument_langfuse_tool()` to wrap tool `invoke`/`ainvoke`, start spans via `core.langgraph`, and emit tool events with input/output metrics.
* Applied instrumentation across coding, file, Excel, and writing tool modules (`src/agents/coding_agent.py`, `src/agents/tools_files.py`, `src/agents/excel_tools.py`, `src/agents/tools_writing.py`, `src/agents/tools.py`, `src/tools/excel_tools.py`, `src/tools/generic_excel_tools.py`).
* Extended `core/langgraph.py` with runtime span/event helpers reused by the tool decorator.

#### 2. **Emit Langfuse events from background `Task` lifecycle**

**✔ Done.**

* Enhanced `Task` (`src/agents/bg_task_agent/task.py`) to accept Langfuse runtime context, capture payload previews, time executions, and emit `start`/`update`/`finish` events through `emit_runtime_event()`.

#### 3. **Background workflow heartbeats**

**✔ Done.**

* Updated `src/agents/bg_task_agent/bg_task_agent.py` to derive runtime context via `build_langfuse_runtime`, feed it into `Task`, and emit heartbeat events (`queued`, `running`, `completed`) during the sample background flow.
* Added regression coverage in `tests/core/test_tools.py` for tool instrumentation and task event emission behaviour.

### ✅ **Step 5 Checklist Review**

#### 1. **Extend `AgentClient` with Langfuse trace/session support**

**✔ Done.**

* Added `_prepare_agent_config()` helper so `invoke`/`ainvoke`/`stream`/`astream` accept `trace_id` & `session_id` kwargs and inject them into the payload (`src/client/client.py`).
* Expanded client docstrings and added regression assertions in `tests/client/test_client.py` verifying the forwarded identifiers.

#### 2. **Persist trace context in Streamlit**

**✔ Done.**

* Established `TRACE_ID_QUERY_PARAM` + `st.session_state.trace_id` lifecycle, generating a UUID when absent and syncing to query params for shareable URLs (`src/streamlit_app.py`).
* Added `update_langfuse_runtime()` to consolidate runtime metadata and ensure every request passes `trace_id`/`session_id` to the API client.

#### 3. **Surface Langfuse trace link in UI**

**✔ Done.**

* Introduced `core.langfuse.build_trace_url()` and reused it from `render_trace_link()` so the chat footer shows a "🔗 View in Langfuse" link bound to the active trace (`src/core/langfuse.py`, `src/streamlit_app.py`).

#### 4. **Feedback & frontend telemetry wiring**

**✔ Done.**

* Feedback submissions now call `AgentClient.acreate_feedback()` with UI metadata while the client handles Langfuse identifiers (`src/streamlit_app.py`, `src/client/client.py`).
* Added lightweight event emitters for page load and file uploads that invoke the Langfuse client directly when tracing is enabled (`src/streamlit_app.py`).

* **✔ Done.** `AgentClient.acreate_feedback` accepts `trace_id`/`session_id`; injects IDs into payload and metadata; tests added.
  * Implemented payload construction and metadata mirroring in `src/client/client.py` with regression coverage in `tests/client/test_client.py`.
* **✔ Done.** Frontend event emission decoupled from `core.langgraph`; uses Langfuse client directly with graceful no-op fallback.
  * Introduced a Streamlit helper that resolves the Langfuse client and no-ops when tracing is disabled (`src/streamlit_app.py`).

### ✅ **Step 6 Checklist Review**

#### 1. **Instrument file lifecycle with Langfuse spans/events**

**✔ Done.**

* Added runtime-aware span and event helpers to every `files_router` endpoint so uploads, listings, downloads, and deletions emit Langfuse spans/events enriched with user/thread/filename metadata (`src/service/files_router.py`).
* Extended client-side file operations to forward trace/session/run identifiers via headers/query params ensuring server spans link to the active trace (`src/client/client.py`, `src/streamlit_app.py`).

#### 2. **Feedback hook records Langfuse scores**

**✔ Done.**

* Feedback endpoint now mirrors submissions to Langfuse by calling `create_score`/`score_current_trace` with comment and metadata fallbacks, reusing payload identifiers when present (`src/service/service.py`).

#### 3. **Return Langfuse reference to clients**

**✔ Done.**

* `FeedbackResponse` exposes `langfuse_trace_id`/`langfuse_run_id`; the client surfaces these values to Streamlit so UI state stays aligned with the recorded trace (`src/schema/schema.py`, `src/client/client.py`, `src/streamlit_app.py`).
* Documented the Feedback API response type and Langfuse header usage in the README and added an integration test that asserts `/files/upload` reads the `X-Langfuse-*` headers (`README.md`, `tests/service/test_service.py`).

### ✅ **Step 7 Checklist Review**

#### 1. **Unit tests for spans, errors, and events**

**✔ Done.**

* Added `tests/service/test_service_langfuse.py` covering `_start_langfuse_span`, `_end_span`, and `_handle_input` with mocked clients/handlers to assert span metadata, error propagation, and callback attachment.
* Extended `tests/core/test_langgraph.py` with `test_emit_runtime_event_includes_context` to verify Langfuse event emission arguments when `get_langfuse_client()` is mocked.

#### 2. **Streaming regression coverage**

**✔ Done.**

* `tests/service/test_service_streaming.py::test_message_generator_includes_langfuse_metadata` asserts SSE payloads include the `langfuse` envelope, `ChatMessage.custom_data` metadata, and that the prepared callbacks flow through to `agent.astream`.

#### 3. **Client & Streamlit trace propagation**

**✔ Done.**

* Updated `tests/app/test_streamlit_app.py` to assert `AgentClient` streaming/invoke calls receive `trace_id`/`session_id`, and that Streamlit session state captures `langfuse_trace_id`, `langfuse_session_id`, and `langfuse_run_id` emitted from SSE metadata.

#### 4. **Documentation refresh**

**✔ Done.**

* README now documents the SSE `langfuse` envelope and metadata embedding, while the new `CONTRIBUTING.md` outlines the Langfuse environment variables required for local development.

#### 5. **CI Langfuse fixtures**

**✔ Done.**

* Default dummy Langfuse credentials are provided via the shared `mock_env` fixture (`tests/conftest.py`) so tests execute without touching the network.

