"""Langfuse instrumentation helpers for LangChain tools."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from core.langgraph import (
    build_langfuse_runtime,
    emit_runtime_event,
    end_runtime_span,
    runtime_metadata,
    start_runtime_span,
    update_runtime_span,
)

MAX_PREVIEW_LENGTH = 400


def _truncate(text: str, limit: int = MAX_PREVIEW_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _coerce_str(value: Any) -> str:
    if isinstance(value, (str, bytes, bytearray)):
        data = value.decode() if isinstance(value, (bytes, bytearray)) else value
        return data
    try:
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)


def _preview(value: Any) -> str | None:
    if value is None:
        return None
    return _truncate(_coerce_str(value))


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    text = _coerce_str(value)
    return len(text)


def _extract_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> RunnableConfig | None:
    if "config" in kwargs and isinstance(kwargs["config"], RunnableConfig):
        return kwargs["config"]
    if len(args) >= 2 and isinstance(args[1], RunnableConfig):
        return args[1]
    return None


def _extract_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("input")


def instrument_langfuse_tool(tool: BaseTool, *, name: str | None = None) -> BaseTool:
    """Wrap ``tool`` so each invocation emits Langfuse telemetry."""

    if getattr(tool, "__langfuse_instrumented__", False):
        return tool

    tool_name = name or getattr(tool, "name", None) or tool.__class__.__name__
    span_name = f"agent.tool.{tool_name}"

    original_invoke = tool.invoke

    def _record(
        runtime: Mapping[str, Any],
        span: Any | None,
        *,
        input_value: Any,
        output_value: Any,
        error: Exception | None,
        started_at: float,
    ) -> None:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        input_size = _payload_size(input_value)
        output_size = _payload_size(output_value) if error is None else 0
        status = "success" if error is None else "error"
        metadata = runtime_metadata(
            runtime,
            tool=tool_name,
            status=status,
            success=error is None,
            duration_ms=round(duration_ms, 3),
            input_size=input_size,
            output_size=output_size if output_size else None,
        )
        metadata["input_preview"] = _preview(input_value)
        if error is None and output_value is not None:
            metadata["output_preview"] = _preview(output_value)
        if error is not None:
            metadata["error"] = f"{error.__class__.__name__}: {error}"

        update_runtime_span(span, metadata=metadata)
        if error is None and output_value is not None:
            update_runtime_span(span, output={"preview": metadata.get("output_preview")})
        end_runtime_span(span, error=error)
        emit_runtime_event(span_name, runtime, metadata=metadata)

    def instrumented_invoke(self: BaseTool, *args: Any, **kwargs: Any) -> Any:
        input_value = _extract_input(args, kwargs)
        config = _extract_config(args, kwargs)
        runtime = build_langfuse_runtime(config=config)
        metadata = runtime_metadata(
            runtime,
            tool=tool_name,
            phase="start",
            input_preview=_preview(input_value),
            input_size=_payload_size(input_value),
        )
        span, context = start_runtime_span(span_name, runtime, metadata=metadata)
        started_at = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            with context:
                result = original_invoke(*args, **kwargs)
            return result
        except Exception as exc:  # pragma: no cover - propagate after logging
            error = exc
            raise
        finally:
            _record(runtime, span, input_value=input_value, output_value=result, error=error, started_at=started_at)

    tool.invoke = instrumented_invoke.__get__(tool, tool.__class__)  # type: ignore[assignment]

    if hasattr(tool, "ainvoke"):
        original_ainvoke = tool.ainvoke  # type: ignore[attr-defined]

        async def instrumented_ainvoke(self: BaseTool, *args: Any, **kwargs: Any) -> Any:
            input_value = _extract_input(args, kwargs)
            config = _extract_config(args, kwargs)
            runtime = build_langfuse_runtime(config=config)
            metadata = runtime_metadata(
                runtime,
                tool=tool_name,
                phase="start",
                input_preview=_preview(input_value),
                input_size=_payload_size(input_value),
            )
            span, context = start_runtime_span(span_name, runtime, metadata=metadata)
            started_at = time.perf_counter()
            error: Exception | None = None
            result: Any = None
            try:
                with context:
                    result = await original_ainvoke(*args, **kwargs)  # type: ignore[call-arg]
                return result
            except Exception as exc:  # pragma: no cover - propagate after logging
                error = exc
                raise
            finally:
                _record(runtime, span, input_value=input_value, output_value=result, error=error, started_at=started_at)

        tool.ainvoke = instrumented_ainvoke.__get__(tool, tool.__class__)  # type: ignore[assignment]

    tool.__langfuse_instrumented__ = True  # type: ignore[attr-defined]
    return tool


__all__ = ["instrument_langfuse_tool"]

