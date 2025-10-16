"""Utilities for managing Langfuse tracing within the service layer."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.client import (  # type: ignore[import-untyped]
    StatefulSpanClient,
    StatefulTraceClient,
)

from core import settings

logger = logging.getLogger(__name__)

_shared_client: Langfuse | None = None
_active_spans: dict[str, StatefulSpanClient] = {}
_run_trace_ids: dict[str, str] = {}


def reset_tracing_state() -> None:
    """Reset cached tracing state (used in unit tests)."""

    global _shared_client
    _active_spans.clear()
    _run_trace_ids.clear()
    _shared_client = None


def set_shared_client(client: Optional[Langfuse]) -> None:
    """Force the shared client (primarily for tests)."""

    global _shared_client
    _shared_client = client


def get_shared_client() -> Langfuse | None:
    """Return a cached Langfuse client if tracing is enabled."""

    global _shared_client
    if not settings.LANGFUSE_TRACING:
        return None
    if _shared_client is None:
        kwargs: dict[str, Any] = {}
        if settings.LANGFUSE_HOST:
            kwargs["host"] = settings.LANGFUSE_HOST
        if settings.LANGFUSE_PUBLIC_KEY:
            kwargs["public_key"] = settings.LANGFUSE_PUBLIC_KEY.get_secret_value()
        if settings.LANGFUSE_SECRET_KEY:
            kwargs["secret_key"] = settings.LANGFUSE_SECRET_KEY.get_secret_value()
        try:
            _shared_client = Langfuse(**kwargs)
        except Exception:
            logger.exception("Failed to initialise Langfuse client")
            return None
    return _shared_client


def start_run_trace(
    *,
    run_id: str,
    trace_id: str,
    agent_id: str,
    user_id: str | None,
    thread_id: str,
    input_message: str,
) -> None:
    """Create Langfuse trace/span records for the beginning of a run."""

    client = get_shared_client()
    if client is None:
        return

    metadata = {"agent_id": agent_id, "thread_id": thread_id, "run_id": run_id}
    try:
        trace_client: StatefulTraceClient = client.trace(
            id=trace_id,
            name=f"agent.invoke.{agent_id}",
            user_id=user_id,
            metadata=metadata,
            input={"message": input_message},
        )
        span_client: StatefulSpanClient = client.span(
            trace_id=trace_id,
            name="service.invoke",
            metadata=metadata,
            input={"message": input_message},
        )
        _active_spans[run_id] = span_client
        _run_trace_ids[run_id] = trace_id
        trace_client.event(
            name="user.message",
            metadata={"message": input_message},
        )
    except Exception:
        logger.exception("Failed to start Langfuse trace for run %s", run_id)


def complete_run_trace(
    *,
    run_id: str,
    output_message: str | None,
    error: str | None,
) -> None:
    """Finalize Langfuse span/event information for a run."""

    trace_id = _run_trace_ids.pop(run_id, None)
    span_client = _active_spans.pop(run_id, None)
    client = get_shared_client()
    if client is None or trace_id is None:
        return

    try:
        if span_client is not None:
            if error:
                span_client.end(
                    output={"error": error},
                    level="ERROR",
                    status_message=error,
                )
            else:
                span_client.end(output={"message": output_message})
    except Exception:
        logger.exception("Failed to end Langfuse span for run %s", run_id)

    try:
        level = "ERROR" if error else "DEFAULT"
        client.event(
            trace_id=trace_id,
            name="agent.response" if not error else "agent.error",
            metadata={
                "run_id": run_id,
                "message": output_message if not error else error,
            },
            level=level,
        )
    except Exception:
        logger.exception("Failed to publish Langfuse event for run %s", run_id)
