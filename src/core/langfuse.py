"""Utilities for interacting with Langfuse tracing."""

from __future__ import annotations

import logging

from langfuse import Langfuse

from core.settings import settings

__all__ = ["get_langfuse_client"]

_logger = logging.getLogger(__name__)

_DISABLED_SENTINEL: bool = False
_LANGFUSE_CLIENT: Langfuse | None = None


def _build_client() -> Langfuse | None:
    """Create a Langfuse client from configured settings."""
    global _DISABLED_SENTINEL

    if not settings.LANGFUSE_TRACING:
        _DISABLED_SENTINEL = True
        return None

    kwargs = {"host": settings.LANGFUSE_HOST}
    if settings.LANGFUSE_PUBLIC_KEY:
        kwargs["public_key"] = settings.LANGFUSE_PUBLIC_KEY.get_secret_value()
    if settings.LANGFUSE_SECRET_KEY:
        kwargs["secret_key"] = settings.LANGFUSE_SECRET_KEY.get_secret_value()

    try:
        return Langfuse(**kwargs)
    except Exception:  # pragma: no cover
        _logger.exception("Failed to initialize Langfuse client")
        _DISABLED_SENTINEL = True
        return None


def get_langfuse_client() -> Langfuse | None:
    """Return a cached Langfuse client if tracing is enabled."""
    global _LANGFUSE_CLIENT, _DISABLED_SENTINEL

    if _DISABLED_SENTINEL:
        return None

    if _LANGFUSE_CLIENT is None:
        _LANGFUSE_CLIENT = _build_client()
        if _LANGFUSE_CLIENT is None:
            return None

    return _LANGFUSE_CLIENT
