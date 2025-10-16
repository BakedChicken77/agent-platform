"""Langfuse instrumentation helpers used across the service."""
from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, TypeVar

from langfuse import Langfuse

from core import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangfuseContext:
    """Snapshot of the active Langfuse trace context."""

    trace_id: str
    parent_observation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_context: ContextVar[LangfuseContext | None] = ContextVar("langfuse_context", default=None)


@lru_cache(maxsize=1)
def _create_client() -> Langfuse | None:
    """Create the Langfuse client if tracing is enabled and credentials exist."""

    if not settings.LANGFUSE_TRACING:
        return None
    try:
        client = Langfuse()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to initialise Langfuse client: %s", exc)
        return None
    if not client.enabled:
        logger.debug("Langfuse client initialised but disabled (missing credentials).")
        return None
    return client


def get_client() -> Langfuse | None:
    """Return a cached Langfuse client if tracing is available."""

    return _create_client()


def get_current_langfuse_context() -> LangfuseContext | None:
    """Return the current context for the running task/tool."""

    return _context.get()


def snapshot_langfuse_context() -> LangfuseContext | None:
    """Return a copy of the current Langfuse context for later reuse."""

    ctx = _context.get()
    if ctx is None:
        return None
    return LangfuseContext(
        trace_id=ctx.trace_id,
        parent_observation_id=ctx.parent_observation_id,
        metadata=dict(ctx.metadata) if ctx.metadata else {},
    )


def _set_context(ctx: LangfuseContext | None) -> Token:
    return _context.set(ctx)


def _reset_context(token: Token) -> None:
    _context.reset(token)


def sanitize_for_observability(value: Any, *, max_length: int = 500) -> Any:
    """Produce a lightweight representation safe for tracing payloads."""

    if isinstance(value, str):
        return value if len(value) <= max_length else value[: max_length - 3] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        limited = list(value.items())[:10]
        return {str(k): sanitize_for_observability(v, max_length=max_length) for k, v in limited}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_observability(v, max_length=max_length) for v in list(value)[:10]]
    return f"<{value.__class__.__name__}>"


class _SpanHandle:
    """Helper for recording metadata and outputs before closing a span."""

    def __init__(self, span) -> None:
        self._span = span
        self._metadata: dict[str, Any] = {}
        self._output: Any = None
        self._has_error = False
        self._error_message: str | None = None

    def add_metadata(self, **metadata: Any) -> None:
        if metadata:
            self._metadata.update(metadata)

    def set_output(self, output: Any) -> None:
        if output is not None:
            self._output = sanitize_for_observability(output)

    def mark_error(self, error: BaseException) -> None:
        self._has_error = True
        self._error_message = str(error)

    def finish(self, duration_ms: float) -> None:
        if not self._span:
            return
        metadata = {"duration_ms": round(duration_ms, 3), **self._metadata}
        if self._has_error:
            self._span.end(metadata=metadata, level="ERROR", status_message=self._error_message)
        else:
            self._span.end(metadata=metadata, output=self._output)


class _NullSpanHandle(_SpanHandle):
    def __init__(self) -> None:
        super().__init__(span=None)

    def add_metadata(self, **metadata: Any) -> None:  # pragma: no cover - noop
        return

    def set_output(self, output: Any) -> None:  # pragma: no cover - noop
        return

    def mark_error(self, error: BaseException) -> None:  # pragma: no cover - noop
        return

    def finish(self, duration_ms: float) -> None:  # pragma: no cover - noop
        return


@contextmanager
def activate_langfuse_run(
    *,
    trace_id: str,
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> _SpanHandle:
    """Context manager that activates a Langfuse trace span for an agent run."""

    client = get_client()
    if not client:
        yield _NullSpanHandle()
        return

    trace = client.trace(
        id=trace_id,
        name=name,
        user_id=user_id,
        session_id=session_id,
        input=sanitize_for_observability(input_payload),
        metadata=sanitize_for_observability(metadata or {}),
    )
    span = trace.span(name=f"{name}:run")
    handle = _SpanHandle(span)
    token = _set_context(
        LangfuseContext(
            trace_id=trace.trace_id,
            parent_observation_id=span.id,
            metadata=metadata or {},
        )
    )
    start = perf_counter()
    try:
        yield handle
    except BaseException as exc:
        handle.mark_error(exc)
        duration = (perf_counter() - start) * 1000
        handle.finish(duration)
        raise
    else:
        duration = (perf_counter() - start) * 1000
        handle.finish(duration)
    finally:
        _reset_context(token)


def log_langfuse_event(
    name: str,
    *,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
    context: LangfuseContext | None = None,
) -> None:
    """Emit a Langfuse event tied to the current (or provided) trace."""

    client = get_client()
    ctx = context or get_current_langfuse_context()
    if not client or ctx is None:
        return
    try:
        client.event(
            name=name,
            trace_id=ctx.trace_id,
            parent_observation_id=ctx.parent_observation_id,
            input=sanitize_for_observability(input_payload),
            metadata=sanitize_for_observability(metadata or {}),
            level=level,
            status_message=status_message,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to emit Langfuse event '%s': %s", name, exc)


F = TypeVar("F", bound=Callable[..., Any])


def _start_tool_span(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[_SpanHandle, float]:
    client = get_client()
    ctx = get_current_langfuse_context()
    if not client or ctx is None:
        return _NullSpanHandle(), perf_counter()
    span = client.span(
        name=name,
        trace_id=ctx.trace_id,
        parent_observation_id=ctx.parent_observation_id,
        input={
            "args": sanitize_for_observability(args),
            "kwargs": sanitize_for_observability(kwargs),
        },
        metadata={"kind": "tool", "tool_name": name},
    )
    return _SpanHandle(span), perf_counter()


def _end_tool_span(handle: _SpanHandle, start_time: float, *, error: BaseException | None, result: Any) -> None:
    duration = (perf_counter() - start_time) * 1000
    if error is not None:
        handle.mark_error(error)
    else:
        handle.set_output(result)
    handle.finish(duration)


def instrument_tool(_func: F | None = None, *, name: str | None = None) -> Callable[[F], F]:
    """Decorator to wrap LangChain tools with Langfuse spans."""

    def decorator(func: F) -> F:
        tool_name = name or getattr(func, "__name__", "tool")

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                handle, start = _start_tool_span(tool_name, args, kwargs)
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    _end_tool_span(handle, start, error=exc, result=None)
                    raise
                _end_tool_span(handle, start, error=None, result=result)
                return result

            return functools.wraps(func)(async_wrapper)  # type: ignore[return-value]

        def sync_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
            handle, start = _start_tool_span(tool_name, args, kwargs)
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                _end_tool_span(handle, start, error=exc, result=None)
                raise
            _end_tool_span(handle, start, error=None, result=result)
            return result

        return functools.wraps(func)(sync_wrapper)  # type: ignore[return-value]

    if _func is not None:
        return decorator(_func)
    return decorator

