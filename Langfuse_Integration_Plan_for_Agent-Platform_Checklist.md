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

