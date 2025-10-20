"""Langfuse helper utilities.

This module wraps optional imports of the Langfuse SDK and exposes
factories for the client, spans, and LangChain callback handlers. The
helpers are resilient to missing dependencies or incomplete
configuration so the service can start even when Langfuse tracing is
not enabled.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache
from typing import Any
from urllib.parse import quote_plus

from core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _resolve_secret(secret: Any) -> str | None:
    from pydantic import SecretStr  # import locally to avoid optional dependency at import time

    if isinstance(secret, SecretStr):
        return secret.get_secret_value()
    if isinstance(secret, str):
        return secret or None
    return None


@lru_cache
def get_langfuse_client() -> Any | None:
    """Return a cached Langfuse client if tracing is enabled.

    Returns ``None`` when tracing is disabled, credentials are missing,
    or the SDK cannot be imported. The helper logs at debug/warning level
    instead of raising so callers can safely opt into Langfuse without
    impacting startup.
    """

    settings: Settings = get_settings()
    if not settings.LANGFUSE_TRACING:
        logger.debug("Langfuse tracing disabled; client will not be created")
        return None

    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Langfuse SDK import failed: %s", exc)
        return None

    secret_key = _resolve_secret(settings.LANGFUSE_SECRET_KEY)
    public_key = _resolve_secret(settings.LANGFUSE_PUBLIC_KEY)

    if not secret_key or not public_key:
        logger.warning(
            "Langfuse tracing requested but keys are missing; tracing will remain disabled"
        )
        return None

    client_kwargs: dict[str, Any] = {
        "host": settings.LANGFUSE_HOST,
        "secret_key": secret_key,
        "public_key": public_key,
    }
    if settings.LANGFUSE_ENVIRONMENT:
        client_kwargs["environment"] = settings.LANGFUSE_ENVIRONMENT
    if settings.LANGFUSE_SAMPLE_RATE is not None:
        client_kwargs["sample_rate"] = settings.LANGFUSE_SAMPLE_RATE

    client = Langfuse(**client_kwargs)
    logger.debug("Langfuse client initialised for host %s", settings.LANGFUSE_HOST)
    return client


def get_langfuse_handler() -> Any | None:
    """Return a LangChain callback handler backed by Langfuse when available."""

    client = get_langfuse_client()
    if client is None:
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Langfuse CallbackHandler import failed: %s", exc)
        return None

    for key in ("client", "langfuse_client"):
        try:
            return CallbackHandler(**{key: client})
        except TypeError:
            continue

    return CallbackHandler()


def create_span(**kwargs: Any) -> AbstractContextManager[Any]:
    """Create a Langfuse span context manager if a client is configured."""

    client = get_langfuse_client()
    if client is None:
        return nullcontext()
    return client.span(**kwargs)


def build_trace_url(trace_id: str | None) -> str | None:
    """Return a Langfuse trace URL for ``trace_id`` when possible."""

    if not trace_id:
        return None

    settings: Settings = get_settings()
    if not settings.LANGFUSE_TRACING:
        return None

    public_key = settings.LANGFUSE_CLIENT_PUBLIC_KEY
    if not public_key:
        return None

    host = settings.LANGFUSE_HOST.rstrip("/")
    url = f"{host}/project/{public_key}/trace/{trace_id}"

    if settings.LANGFUSE_ENVIRONMENT:
        url = f"{url}?env={quote_plus(settings.LANGFUSE_ENVIRONMENT)}"

    return url
