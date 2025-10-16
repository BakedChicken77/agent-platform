"""Helpers for instrumenting LangGraph agents with Langfuse spans."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

from langchain_core.runnables import RunnableConfig

from core.langfuse import get_langfuse_client

if TYPE_CHECKING:  # pragma: no cover
    from agents.agent_registry import AgentGraph

__all__ = [
    "configure_agent",
    "get_agent_metadata",
    "with_langfuse_span",
]

logger = logging.getLogger(__name__)

_AGENT_METADATA_ATTR = "_agent_metadata"

T = TypeVar("T")


def _extract_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> RunnableConfig | None:
    cfg = kwargs.get("config")
    if isinstance(cfg, dict) and "configurable" in cfg:
        return cast(RunnableConfig, cfg)

    for arg in reversed(args):
        if isinstance(arg, dict) and "configurable" in arg:
            return cast(RunnableConfig, arg)
    return None


def _start_span(node_name: str, config: RunnableConfig | None):
    if not config:
        return None

    configurable = config.get("configurable") or {}
    trace_id = configurable.get("trace_id")
    if not trace_id:
        return None

    client = get_langfuse_client()
    if client is None:
        return None

    session_id = configurable.get("session_id")
    metadata = {
        "agent_id": configurable.get("agent_id"),
        "agent_kind": configurable.get("agent_kind"),
        "node": node_name,
    }

    try:
        return client.span(
            name=node_name,
            trace_id=trace_id,
            session_id=session_id,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
    except Exception:  # pragma: no cover
        logger.exception("Failed to create Langfuse span for node '%s'", node_name)
        return None


def with_langfuse_span(node_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                config = _extract_config(args, kwargs)
                span = _start_span(node_name, config)
                try:
                    return await cast(Awaitable[T], func(*args, **kwargs))
                except Exception as exc:
                    if span is not None:
                        try:
                            span.update(level="ERROR", status_message=str(exc))
                        except Exception:  # pragma: no cover
                            logger.exception(
                                "Failed to update Langfuse span for node '%s'", node_name
                            )
                    raise
                finally:
                    if span is not None:
                        try:
                            span.end()
                        except Exception:  # pragma: no cover
                            logger.exception(
                                "Failed to close Langfuse span for node '%s'", node_name
                            )

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            config = _extract_config(args, kwargs)
            span = _start_span(node_name, config)
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if span is not None:
                    try:
                        span.update(level="ERROR", status_message=str(exc))
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "Failed to update Langfuse span for node '%s'", node_name
                        )
                raise
            finally:
                if span is not None:
                    try:
                        span.end()
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "Failed to close Langfuse span for node '%s'", node_name
                        )

        return sync_wrapper

    return decorator


def configure_agent(
    graph: AgentGraph,
    *,
    agent_id: str,
    agent_kind: str,
    description: str | None = None,
) -> AgentGraph:
    metadata = {
        "agent_id": agent_id,
        "agent_kind": agent_kind,
    }
    if description:
        metadata["description"] = description

    instrumented = graph.with_config(
        metadata=metadata,
        configurable={
            "agent_id": agent_id,
            "agent_kind": agent_kind,
            "trace_id": None,
            "session_id": None,
        },
    )

    for attr in ("checkpointer", "store"):
        if hasattr(graph, attr):
            setattr(instrumented, attr, getattr(graph, attr))

    if hasattr(graph, "name"):
        instrumented.name = getattr(graph, "name")

    setattr(instrumented, _AGENT_METADATA_ATTR, metadata)
    return instrumented


def get_agent_metadata(agent: Any) -> dict[str, Any]:
    base = getattr(agent, _AGENT_METADATA_ATTR, {}) or {}
    if hasattr(agent, "name") and "agent_id" not in base:
        base = {**base, "agent_id": getattr(agent, "name")}
    return base
