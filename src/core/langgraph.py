"""LangGraph instrumentation helpers for Langfuse telemetry."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import AbstractContextManager, nullcontext
from functools import wraps
from typing import Any

from langchain_core.runnables import RunnableConfig

from core.langfuse import get_langfuse_client

logger = logging.getLogger(__name__)

LANGFUSE_STATE_KEY = "langfuse"
LANGFUSE_CONFIG_KEY = "langfuse"


def _normalise_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    return None


def _merge_dicts(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        if not source:
            continue
        for key, value in source.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def _langfuse_from_config(config: RunnableConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    configurable = config.get("configurable", {}) or {}
    nested = configurable.get(LANGFUSE_CONFIG_KEY)
    runtime = dict(nested) if isinstance(nested, Mapping) else {}
    for key in ("trace_id", "session_id", "run_id", "agent_name", "user_id", "thread_id"):
        value = configurable.get(key)
        if value is not None and key not in runtime:
            runtime[key] = value
    return runtime


def _langfuse_from_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    nested = state.get(LANGFUSE_STATE_KEY)
    if isinstance(nested, Mapping):
        return dict(nested)
    return {}


def build_langfuse_runtime(
    *,
    config: RunnableConfig | None = None,
    state: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge Langfuse runtime details from config/state/previous data."""

    return _merge_dicts(
        _langfuse_from_state(previous),
        _langfuse_from_state(state),
        _langfuse_from_config(config),
    )


def ensure_langfuse_state(
    container: MutableMapping[str, Any],
    *,
    config: RunnableConfig | None = None,
    state: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Persist merged Langfuse runtime data into ``container`` in-place."""

    runtime = build_langfuse_runtime(config=config, state=state, previous=previous)
    if runtime:
        container[LANGFUSE_STATE_KEY] = runtime
    return container


def persist_langfuse_state(
    payload: Mapping[str, Any],
    *,
    config: RunnableConfig | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``payload`` with Langfuse runtime data attached."""

    data = dict(payload)
    ensure_langfuse_state(data, config=config, previous=previous)
    return data


def _start_node_span(
    node_name: str,
    *,
    config: RunnableConfig | None = None,
    state: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> tuple[Any | None, AbstractContextManager[Any]]:
    runtime = build_langfuse_runtime(config=config, state=state, previous=previous)
    client = get_langfuse_client()
    if not runtime or client is None:
        return None, nullcontext()
    try:
        span_kwargs: dict[str, Any] = {
            "name": f"agent.node.{node_name}",
            "metadata": {
                "agent_name": runtime.get("agent_name"),
                "run_id": runtime.get("run_id"),
                "thread_id": runtime.get("thread_id"),
                "user_id": runtime.get("user_id"),
                "node": node_name,
            },
        }
        if runtime.get("trace_id"):
            span_kwargs["trace_id"] = runtime["trace_id"]
        if runtime.get("session_id"):
            span_kwargs["session_id"] = runtime["session_id"]
        span = client.span(**span_kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to create Langfuse node span for %s: %s", node_name, exc)
        return None, nullcontext()

    starter = getattr(span, "start_as_current_span", None)
    if callable(starter):
        try:
            context = starter()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to start Langfuse span context for %s: %s", node_name, exc)
            context = nullcontext()
    else:
        context = nullcontext()
    return span, context


def _finish_node_span(span: Any | None, error: Exception | None = None) -> None:
    if span is None:
        return
    try:
        if error is None:
            span.end()
        else:
            span.end(level="ERROR", status_message=f"{error.__class__.__name__}: {error}")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to end Langfuse node span: %s", exc)


def _find_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> RunnableConfig | None:
    config = kwargs.get("config")
    if isinstance(config, RunnableConfig):
        return config
    for value in reversed(args):
        if isinstance(value, RunnableConfig):
            return value
    return None


def _find_state(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Mapping[str, Any] | None:
    for value in args:
        mapping = _normalise_mapping(value)
        if mapping is not None:
            return mapping
    for value in kwargs.values():
        mapping = _normalise_mapping(value)
        if mapping is not None:
            return mapping
    return None


def _wrap_callable(node: Callable[..., Any], node_name: str) -> Callable[..., Any]:
    if asyncio.iscoroutinefunction(node):
        
        @wraps(node)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = _find_config(args, kwargs)
            state = _find_state(args, kwargs)
            span, context = _start_node_span(node_name, config=config, state=state)
            error: Exception | None = None
            try:
                with context:
                    result = await node(*args, **kwargs)
                return result
            except Exception as exc:  # pragma: no cover - propagate after logging
                error = exc
                raise
            finally:
                _finish_node_span(span, error)

        async_wrapper.__langfuse_instrumented__ = True  # type: ignore[attr-defined]
        return async_wrapper

    @wraps(node)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        config = _find_config(args, kwargs)
        state = _find_state(args, kwargs)
        span, context = _start_node_span(node_name, config=config, state=state)
        error: Exception | None = None
        try:
            with context:
                return node(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - propagate after logging
            error = exc
            raise
        finally:
            _finish_node_span(span, error)

    sync_wrapper.__langfuse_instrumented__ = True  # type: ignore[attr-defined]
    return sync_wrapper


class _RunnableWrapper:
    def __init__(self, inner: Any, node_name: str) -> None:
        self._inner = inner
        self._node_name = node_name
        self.__langfuse_instrumented__ = True
        if hasattr(inner, "name"):
            self.name = getattr(inner, "name")

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, item)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.get("config") if kwargs else None
        state = args[0] if args else None
        span, context = _start_node_span(
            self._node_name,
            config=config if isinstance(config, RunnableConfig) else None,
            state=_normalise_mapping(state),
        )
        error: Exception | None = None
        try:
            with context:
                return self._inner.invoke(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - propagate after logging
            error = exc
            raise
        finally:
            _finish_node_span(span, error)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.get("config") if kwargs else None
        state = args[0] if args else None
        span, context = _start_node_span(
            self._node_name,
            config=config if isinstance(config, RunnableConfig) else None,
            state=_normalise_mapping(state),
        )
        error: Exception | None = None
        try:
            with context:
                return await self._inner.ainvoke(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - propagate after logging
            error = exc
            raise
        finally:
            _finish_node_span(span, error)


def instrument_langgraph_node(node: Any | None = None, node_name: str | None = None) -> Any:
    """Return a Langfuse-instrumented version of ``node``."""

    if isinstance(node, str) and node_name is None:
        node, node_name = None, node

    if node is None:
        if node_name is None:
            raise TypeError("node_name is required when using instrument_langgraph_node as a decorator")

        def decorator(actual_node: Any) -> Any:
            return instrument_langgraph_node(actual_node, node_name)

        return decorator

    if node_name is None:
        raise TypeError("node_name must be provided when instrumenting a node")

    if getattr(node, "__langfuse_instrumented__", False):
        return node
    if hasattr(node, "ainvoke") and hasattr(node, "invoke"):
        return _RunnableWrapper(node, node_name)
    return _wrap_callable(node, node_name)


def runtime_metadata(runtime: Mapping[str, Any] | None, **extra: Any) -> dict[str, Any]:
    """Return Langfuse metadata derived from runtime context and ``extra`` values."""

    metadata: dict[str, Any] = {}
    if runtime:
        for key in ("agent_name", "run_id", "thread_id", "user_id"):
            value = runtime.get(key)
            if value is not None:
                metadata[key] = value
    for key, value in extra.items():
        if value is not None:
            metadata[key] = value
    return metadata


def start_runtime_span(
    name: str,
    runtime: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Any | None, AbstractContextManager[Any]]:
    """Start a Langfuse span bound to ``runtime`` context if possible."""

    client = get_langfuse_client()
    if client is None or not runtime:
        return None, nullcontext()

    span_kwargs: dict[str, Any] = {"name": name}
    if metadata:
        span_kwargs["metadata"] = dict(metadata)

    for key in ("trace_id", "session_id", "user_id"):
        value = runtime.get(key)
        if value is not None:
            span_kwargs[key] = value

    try:
        span = client.span(**span_kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to create Langfuse runtime span %s: %s", name, exc)
        return None, nullcontext()

    starter = getattr(span, "start_as_current_span", None)
    if callable(starter):
        try:
            context = starter()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to enter Langfuse runtime span %s: %s", name, exc)
            context = nullcontext()
    else:
        context = nullcontext()

    return span, context


def update_runtime_span(span: Any | None, **payload: Any) -> None:
    """Update ``span`` with additional payload data when available."""

    if span is None or not payload:
        return
    try:
        span.update(**payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to update Langfuse runtime span: %s", exc)


def end_runtime_span(span: Any | None, error: Exception | None = None) -> None:
    """End ``span`` gracefully, marking failures when ``error`` is provided."""

    if span is None:
        return
    try:
        if error is None:
            span.end()
        else:
            span.end(level="ERROR", status_message=f"{error.__class__.__name__}: {error}")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to end Langfuse runtime span: %s", exc)


def emit_runtime_event(
    name: str,
    runtime: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit a Langfuse event associated with ``runtime`` context."""

    if not runtime:
        return

    client = get_langfuse_client()
    if client is None:
        return

    emitter = getattr(client, "event", None)
    if not callable(emitter):  # pragma: no cover - optional API
        return

    event_kwargs: dict[str, Any] = {"name": name}
    if metadata:
        event_kwargs["metadata"] = dict(metadata)

    for key in ("trace_id", "session_id", "user_id"):
        value = runtime.get(key)
        if value is not None:
            event_kwargs[key] = value

    try:
        emitter(**event_kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to emit Langfuse runtime event %s: %s", name, exc)
