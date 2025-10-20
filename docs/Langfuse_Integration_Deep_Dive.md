---
title: "Langfuse Integration Deep Dive"
summary: "Detailed review of the Agent-Platform Langfuse observability pipeline, covering architecture, runtime behaviour, testing, and developer guidance."
---

# Langfuse Integration Deep Dive

- [Overview](#overview)
- [Integration Architecture](#integration-architecture)
- [Runtime Behaviour](#runtime-behaviour)
- [Testing Strategy](#testing-strategy)
- [Developer Setup & Debugging](#developer-setup--debugging)
- [Future Extensions](#future-extensions)

## Overview

Langfuse provides unified tracing, scoring, and observability for LLM applications. The Agent-Platform integration follows a seven-step plan that spans configuration, backend tracing, LangGraph instrumentation, tool telemetry, client linkage, file lifecycle coverage, and documentation.【F:Langfuse_Integration_Plan_for_Agent-Platform.md†L1-L170】 Together these steps deliver full-stack visibility so developers can debug agent behaviour, correlate frontend sessions with backend runs, and capture user feedback alongside traces.【F:README.md†L92-L133】

## Integration Architecture

```text
Streamlit UI
  │  (trace/session propagation, feedback, file uploads)
  ▼
AgentClient (headers + agent_config langfuse payload)
  │
FastAPI service (trace bootstrap → spans → SSE metadata)
  │
LangGraph nodes & tools (node spans, runtime events, background tasks)
  │
File router + ancillary services (runtime spans/events for lifecycle)
  ▼
Langfuse backend (trace/run/session IDs, scores, events)
```

### Core helper (`core/langfuse.py`)

* Lazy client factory respects `LANGFUSE_TRACING` and missing credentials so deployments without Langfuse still start cleanly.【F:src/core/langfuse.py†L33-L75】
* Helper functions expose LangChain callback handlers, span contexts, and public trace URL generation for the Streamlit link.【F:src/core/langfuse.py†L78-L129】

### FastAPI service telemetry (`src/service/service.py`)

* `_prepare_langfuse_context` establishes per-request telemetry (trace, session, user) and reuses handlers when available.【F:src/service/service.py†L200-L248】
* Span helpers capture lifecycle updates, propagate errors, and annotate SSE streaming for messages, tokens, and errors.【F:src/service/service.py†L251-L400】【F:src/service/service.py†L640-L768】
* `_handle_input` injects Langfuse metadata into `RunnableConfig` and LangGraph state so downstream nodes inherit the trace context.【F:src/service/service.py†L428-L520】
* Feedback submissions mirror scores to LangSmith and Langfuse, returning identifiers to the client for UI acknowledgement.【F:src/service/service.py†L810-L863】
* `/health` caches Langfuse connectivity status to avoid repeated network checks.【F:src/service/service.py†L402-L426】

### LangGraph runtime hooks (`core/langgraph.py` & agents)

* `instrument_langgraph_node` wraps nodes with spans that inherit trace/session IDs and captures success or error termination.【F:src/core/langgraph.py†L33-L207】
* Runtime helpers persist trace metadata across checkpoints, emit events, and start spans for arbitrary background work.【F:src/core/langgraph.py†L209-L406】
* Agents such as the coding expert ensure Langfuse state is persisted and nodes are decorated, keeping context intact across graph transitions.【F:src/agents/coding_agent.py†L29-L233】

### Tool and task telemetry (`core/tools.py`, `agents/bg_task_agent/task.py`)

* `instrument_langfuse_tool` wraps LangChain tools, recording previews, payload sizes, timings, and emitting span + event updates.【F:src/core/tools.py†L1-L182】
* Background `Task` emits structured start/update/finish events with runtime metadata, including durations and payload previews.【F:src/agents/bg_task_agent/task.py†L32-L106】

### File lifecycle instrumentation (`src/service/files_router.py`)

* Request handlers derive trace context from headers or query parameters and propagate it through spans/events for uploads, listings, downloads, and deletions.【F:src/service/files_router.py†L86-L448】

### Client and Streamlit linkage (`src/client/client.py`, `src/streamlit_app.py`)

* `AgentClient` forwards trace/session IDs through headers and `agent_config.langfuse`, ensuring API calls and SSE streams are stitched to the active trace.【F:src/client/client.py†L74-L307】
* Streamlit stores trace metadata in session state, pushes frontend events, hydrates IDs from streamed messages, and renders a “View in Langfuse” link when configuration allows.【F:src/streamlit_app.py†L173-L271】

## Runtime Behaviour

### Telemetry flow

1. Streamlit generates or restores a `trace_id` and threads it through API calls alongside the thread/session identifiers.【F:src/streamlit_app.py†L173-L220】
2. `AgentClient` embeds these identifiers into headers and the LangChain config payload sent to FastAPI.【F:src/client/client.py†L74-L307】
3. `_handle_input` merges the metadata into `RunnableConfig` and attaches Langfuse callbacks before invoking the LangGraph agent.【F:src/service/service.py†L428-L520】
4. Span contexts wrap `agent.ainvoke`, SSE streaming, and downstream node/tool execution, emitting events and span updates as work progresses.【F:src/service/service.py†L251-L400】【F:src/core/langgraph.py†L33-L406】
5. SSE responses deliver Langfuse identifiers inside both the top-level event payload and each `ChatMessage.custom_data['langfuse']`, allowing the UI to capture run/session updates in real time.【F:src/service/service.py†L640-L747】
6. Streamlit stores the streamed identifiers and updates the cached runtime context, keeping future actions correlated with the same trace.【F:src/streamlit_app.py†L227-L260】

### Example SSE payload

```json
{
  "type": "message",
  "content": {
    "type": "ai",
    "run_id": "1c4...",
    "custom_data": {
      "langfuse": {
        "trace_id": "trace-xyz",
        "session_id": "session-xyz",
        "run_id": "1c4...",
        "user_id": "user-42"
      }
    }
  },
  "langfuse": {
    "trace_id": "trace-xyz",
    "session_id": "session-xyz",
    "run_id": "1c4...",
    "user_id": "user-42"
  }
}
```

This structure matches the payload constructed in `message_generator`, which injects the same identifiers into both the event wrapper and the message body.【F:src/service/service.py†L676-L704】

### Disabled tracing fallback

When `LANGFUSE_TRACING` is false or credentials are missing, Langfuse helpers return `None`/`nullcontext`, so all instrumentation becomes a no-op without raising errors.【F:src/core/langfuse.py†L33-L129】 Regression tests verify that the client/handler are skipped when tracing is disabled or the SDK is absent.【F:tests/core/test_langfuse.py†L34-L104】

## Testing Strategy

The following suites provide regression coverage for the integration:

* **Core helpers** – `tests/core/test_langfuse.py` validates client bootstrap, fallback behaviour, and handler creation under mocked environments.【F:tests/core/test_langfuse.py†L34-L104】
* **LangGraph instrumentation** – `tests/core/test_langgraph.py` checks node spans, state persistence, and runtime event metadata.【F:tests/core/test_langgraph.py†L39-L113】
* **Tool and task telemetry** – `tests/core/test_tools.py` asserts tool spans/events and task lifecycle emissions while handling the no-client case gracefully.【F:tests/core/test_tools.py†L68-L128】
* **FastAPI service** – `tests/service/test_service_langfuse.py` covers span lifecycles and callback injection, while `tests/service/test_service_streaming.py` ensures SSE payloads carry Langfuse metadata.【F:tests/service/test_service_langfuse.py†L50-L134】【F:tests/service/test_service_streaming.py†L67-L118】
* **File router** – `tests/service/test_service.py` verifies Langfuse headers and runtime propagation on file uploads.【F:tests/service/test_service.py†L260-L319】
* **Streamlit client** – `tests/app/test_streamlit_app.py` validates that session state captures trace metadata and that the client forwards IDs on streaming requests.【F:tests/app/test_streamlit_app.py†L38-L165】
* **CI fixtures** – `tests/conftest.py` seeds dummy Langfuse environment variables to keep tests isolated from real credentials.【F:tests/conftest.py†L20-L36】

> **Note:** Running `pytest` without the optional LangChain/Streamlit dependencies results in collection failures for unrelated modules. Install the full dev requirements (see [Developer Setup & Debugging](#developer-setup--debugging)) before executing the suite.【8c76b9†L1-L74】

## Developer Setup & Debugging

1. **Configure environment variables** – Copy `.env.example` and supply Langfuse keys alongside required model credentials.【F:README.md†L92-L123】
2. **Local services** – Start FastAPI and Streamlit via `uv run` to pick up environment settings.【F:CONTRIBUTING.md†L32-L36】
3. **Enable/disable tracing** – Toggle `LANGFUSE_TRACING` in `.env` to switch between full observability and no-op mode.【F:README.md†L92-L123】
4. **Run Langfuse-specific tests** – Execute `uv run pytest tests/core/test_langfuse.py tests/core/test_langgraph.py tests/service/test_service_langfuse.py` after installing dependencies with `uv sync`. This subset exercises the telemetry pipeline without requiring the full suite.【F:CONTRIBUTING.md†L9-L47】
5. **Health checks & dashboards** – Hit `/health` to confirm Langfuse connectivity, then inspect traces using the generated link in the Streamlit footer.【F:src/service/service.py†L402-L426】【F:src/streamlit_app.py†L263-L271】
6. **Debugging tips** – Review SSE payloads (they include trace/session/run IDs) and Langfuse events emitted from tools, background tasks, and file operations for contextual metadata.【F:src/service/service.py†L640-L768】【F:src/core/tools.py†L78-L182】【F:src/agents/bg_task_agent/task.py†L32-L106】【F:src/service/files_router.py†L86-L448】

## Future Extensions

* **Dashboard integrations** – Surface Langfuse trace summaries inside Streamlit or a dedicated admin panel for quick triage.【F:src/streamlit_app.py†L263-L271】
* **Analytics aggregation** – Combine Langfuse events with LangSmith data to build performance heatmaps or time-series alerts.【F:src/service/service.py†L810-L858】
* **Automated sampling controls** – Extend settings to dynamically adjust `LANGFUSE_SAMPLE_RATE` per agent or workload stage.【F:src/core/langfuse.py†L63-L75】
* **Custom frontend telemetry** – Expand `emit_frontend_event` usage to capture UI interactions (button clicks, latency metrics) and correlate them with backend spans.【F:src/streamlit_app.py†L195-L224】

---

_For additional background, review the original [Langfuse Integration Plan](../Langfuse_Integration_Plan_for_Agent-Platform.md) and the accompanying checklist to understand how each milestone maps to the implementation._
