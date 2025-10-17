# Langfuse Integration Plan for Agent-Platform

This document outlines a 7-step roadmap to integrate **Langfuse** observability across the Agent-Platform stack. Follow these steps in order to enable comprehensive telemetry, tracing, and debugging visibility for developers and end users.

---

## Step 1: Observability Bootstrap & Configuration

Establish core Langfuse setup across both API and UI containers.

### Tasks:

* ✅ **Create `core/langfuse.py`**

  * Add a lazily initialized singleton/factory that wraps the `Langfuse` client.
  * Expose helpers like `get_client()`, `get_span()`, `capture_exception()`, etc.
  * Respect `settings.LANGFUSE_TRACING` to toggle all activity.

* ✅ **Extend `Settings` and `.env.example`**

  * Add:

    * `LANGFUSE_PUBLIC_KEY`
    * `LANGFUSE_SECRET_KEY`
    * `LANGFUSE_HOST`
    * `LANGFUSE_SAMPLING_RATE` (optional)
    * `LANGFUSE_ENVIRONMENT`
    * `LANGFUSE_TRACING` (boolean flag)

* ✅ **Update Compose and Dockerfiles**

  * In `compose.yaml`, add `LANGFUSE_*` under `environment` and `build.args`.
  * Modify both `Dockerfile.api` and `Dockerfile.streamlit` to accept these `ARG`s.

* ✅ **Guard Execution and Document Usage**

  * Ensure app runs cleanly with tracing off.
  * Add `README.md` section: “🔍 Langfuse Observability: How to Enable”

---

## Step 2: FastAPI Request/Response Tracing

Instrument API entrypoints with Langfuse spans and metadata.

### Tasks:

* ✅ In `service.py`, wrap:

  * `agent.ainvoke()` and `agent.astream()` with `start_as_current_span()`.
  * Tag with metadata: `agent_id`, `thread_id`, `user_id`.

* ✅ Populate `RunnableConfig.callbacks` with a Langfuse handler scoped to:

  * `trace_id`, `session_id`, `user_id`.

* ✅ Exception Handling:

  * Use `capture_exception()` or span status to report errors gracefully.
  * Ensure full trace remains usable on exceptions.

* ✅ Update `message_generator`:

  * Log SSE streaming as child spans.
  * Report start/finish/token yields for deep inspection.

* ✅ Optimize `/health`:

  * Use shared `get_client()` and cache the connectivity result.

---

## Step 3: LangGraph Workflow Metadata

Propagate trace and session context through LangGraph DAGs.

### Tasks:

* ✅ Modify each `AgentGraph` instance:

  * `chatbot_graph`, `supervisor_graph`, `coding_graph`, `bg_task_graph`
  * Inject `trace_id`, `session_id`, `agent_name` via `.with_config()` or `RunnableConfig.configurable`.

* ✅ Node-Level Span Decoration:

  * Wrap each node with a decorator that creates child spans using node names for tree-structured tracing.

* ✅ Persist Langfuse IDs:

  * Store `trace_id`, `run_id`, and `session_id` in checkpoint/resume data to preserve continuity across interrupts.

---

## Step 4: Tool and Background Task Telemetry

Emit Langfuse events from tools and long-running background logic.

### Tasks:

* ✅ Instrument tools:

  * Wrap `python_repl()` and any `@tool` methods to record:

    * Input snippet
    * Duration
    * Success/failure
    * Output payload size

* ✅ Emit events in `Task`:

  * On `start()`, `write_data()`, `finish()` log Langfuse events using task context.

* ✅ Background Heartbeats:

  * For long-running workflows, emit periodic heartbeat events (e.g., `status: queued`, `status: completed`) tied to trace context.

---

## Step 5: Client & Streamlit Session Linkage

Ensure trace/session IDs flow between frontend and backend.

### Tasks:

* ✅ Extend `AgentClient`:

  * Accept `trace_id` and `session_id` as optional arguments.
  * Forward them via request headers or `agent_config`.

* ✅ Store Trace in Streamlit:

  * Save `trace_id` in `st.session_state` along with `thread_id`.

* ✅ Display Trace:

  * After agent response, show a “🔗 View in Langfuse” link to trace URL.

* ✅ Feedback Integration:

  * Include trace/run IDs when submitting feedback.
  * Emit optional frontend events (e.g., `page_load`, `file_upload`) using a lightweight Langfuse client.

---

## Step 6: File Lifecycle & Feedback Scoring

Track file operations and score feedback in Langfuse.

### Tasks:

* ✅ In `files_router`:

  * Log span or event for every:

    * Upload
    * Download
    * Delete
    * List
  * Include metadata: `user_id`, `thread_id`, `filename`, `file_size`.

* ✅ Feedback Hook:

  * Submit scores to **both** LangSmith and Langfuse:

    * LangSmith for legacy use
    * `langfuse.score_current_trace()` for new traces

* ✅ Return Reference:

  * Respond with Langfuse trace/run ID so client can acknowledge it visually.

---

## Step 7: Regression Coverage & Documentation

Ensure testability and maintain updated documentation.

### Tasks:

* ✅ Unit Tests:

  * Mock `langfuse.get_client()` and test:

    * Span creation
    * Exception capture
    * Event logging
  * Ensure no network calls in CI.

* ✅ Streaming Tests:

  * Verify SSE includes `trace_id` and that callbacks are used properly.

* ✅ Client/Streamlit Tests:

  * Check `session_state` for trace metadata.
  * Validate propagation to `AgentClient`.

* ✅ Update Docs:

  * Include Langfuse variables in:

    * `.env.example`
    * `README.md`
    * `CONTRIBUTING.md` (for dev setup)

* ✅ CI Fixture:

  * Populate dummy `LANGFUSE_*` values in test environment.

---

### Final Notes

* 🟢 This plan is modular—each step builds on the previous and can be implemented independently.
* 🔁 After completing each step, validate by running sample traces via the frontend and inspecting them in the Langfuse dashboard.
* 📌 This document should remain in the repo (`docs/` or `integration/`) as a reference for ongoing observability maintenance.
