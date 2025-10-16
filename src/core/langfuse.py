"""Utilities for interacting with Langfuse across the application."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID

import langfuse as langfuse_module
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
from langfuse.client import (  # type: ignore[import-untyped]
    StatefulSpanClient,
    StatefulTraceClient,
)

from core.settings import get_settings

__all__ = ["RequestSpanContext", "get_langfuse_client", "cached_auth_check", "request_span"]

logger = logging.getLogger(__name__)


@dataclass
class RequestSpanContext:
    """Wraps a Langfuse span/trace pair with helper utilities."""

    trace: StatefulTraceClient
    span: StatefulSpanClient
    handler: CallbackHandler
    metadata: dict[str, Any]

    @property
    def trace_id(self) -> str:
        return self.trace.trace_id

    def event(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Record a child observation for the request span."""

        try:
            self.span.event(
                name=name,
                metadata=metadata,
                input=input,
                output=output,
                level=level,
                status_message=status_message,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to log Langfuse event %s: %s", name, exc)

    def set_output(self, output: Any) -> None:
        """Attach output payload to the span/trace."""

        try:
            self.span.update(output=output)
            self.trace.update(output=output)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to update Langfuse span output: %s", exc)

    def record_exception(self, error: BaseException) -> None:
        """Record an exception on the span and forward to Langfuse if supported."""

        message = str(error)
        try:
            self.span.update(level="ERROR", status_message=message)
            self.trace.update(level="ERROR", status_message=message)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to update Langfuse span status: %s", exc)

        capture_exception = getattr(langfuse_module, "capture_exception", None)
        if callable(capture_exception):
            try:
                capture_exception(error)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to forward exception to Langfuse: %s", exc)


@lru_cache(maxsize=1)
def _get_client() -> Langfuse:
    settings = get_settings()
    kwargs: dict[str, Any] = {"host": settings.LANGFUSE_HOST}
    if settings.LANGFUSE_PUBLIC_KEY:
        kwargs["public_key"] = settings.LANGFUSE_PUBLIC_KEY.get_secret_value()
    if settings.LANGFUSE_SECRET_KEY:
        kwargs["secret_key"] = settings.LANGFUSE_SECRET_KEY.get_secret_value()
    return Langfuse(**kwargs)


def get_langfuse_client() -> Langfuse | None:
    """Return a shared Langfuse client instance when tracing is enabled."""

    if not get_settings().LANGFUSE_TRACING:
        return None
    try:
        return _get_client()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Unable to initialise Langfuse client: %s", exc)
        return None


@lru_cache(maxsize=1)
def cached_auth_check() -> bool:
    """Cached Langfuse authentication check to avoid repeated round-trips."""

    client = get_langfuse_client()
    if client is None:
        return False
    try:
        return bool(client.auth_check())
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Langfuse auth check failed: %s", exc)
        return False


@contextmanager
def request_span(
    *,
    agent_id: str,
    thread_id: str,
    user_id: str | None,
    run_id: UUID | None = None,
    input_payload: Any | None = None,
) -> Generator[RequestSpanContext | None, None, None]:
    """Context manager yielding a RequestSpanContext when Langfuse is enabled."""

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    metadata: dict[str, Any] = {"agent_id": agent_id, "thread_id": thread_id}
    if user_id:
        metadata["user_id"] = user_id
    if run_id is not None:
        metadata["run_id"] = str(run_id)

    trace = client.trace(
        name=f"{agent_id}.request",
        user_id=user_id or None,
        session_id=thread_id,
        metadata=metadata,
        input=input_payload,
    )
    span = trace.span(name="agent.graph", metadata=metadata)
    handler = CallbackHandler(
        stateful_client=span,
        session_id=thread_id,
        user_id=user_id or None,
        metadata=metadata,
    )

    context = RequestSpanContext(trace=trace, span=span, handler=handler, metadata=metadata)
    try:
        yield context
    except Exception as exc:
        context.record_exception(exc)
        raise
    finally:
        try:
            span.end()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to close Langfuse span: %s", exc)
