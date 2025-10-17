import builtins
import contextlib
import importlib
import os
import sys
import types

import pytest

# Ensure an LLM key is always present before core.settings is imported
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

langfuse = importlib.import_module("core.langfuse")
settings_module = importlib.import_module("core.settings")


@pytest.fixture(autouse=True)
def reset_langfuse_state(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_CLIENT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_ENVIRONMENT", raising=False)
    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE", raising=False)
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)
    monkeypatch.delitem(sys.modules, "langfuse.callback", raising=False)
    langfuse.get_langfuse_client.cache_clear()
    settings_module.get_settings.cache_clear()
    yield
    langfuse.get_langfuse_client.cache_clear()
    settings_module.get_settings.cache_clear()


def test_client_not_created_when_tracing_disabled():
    client = langfuse.get_langfuse_client()
    handler = langfuse.get_langfuse_handler()
    span = langfuse.create_span(name="noop")

    assert client is None
    assert handler is None
    assert isinstance(span, contextlib.AbstractContextManager)


def test_client_missing_sdk(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.5")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ModuleNotFoundError("No module named 'langfuse'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = langfuse.get_langfuse_client()
    handler = langfuse.get_langfuse_handler()

    assert client is None
    assert handler is None


def test_client_and_handler_created(monkeypatch):
    class DummyLangfuse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def span(self, **kwargs):
            return ("span", kwargs)

        def auth_check(self):
            return True

    class DummyCallbackHandler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    dummy_module = types.ModuleType("langfuse")
    dummy_module.Langfuse = DummyLangfuse
    dummy_callback = types.ModuleType("langfuse.callback")
    dummy_callback.CallbackHandler = DummyCallbackHandler

    monkeypatch.setitem(sys.modules, "langfuse", dummy_module)
    monkeypatch.setitem(sys.modules, "langfuse.callback", dummy_callback)

    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_ENVIRONMENT", "staging")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")

    client = langfuse.get_langfuse_client()
    assert isinstance(client, DummyLangfuse)
    assert client.kwargs["environment"] == "staging"
    assert client.kwargs["sample_rate"] == 0.25

    handler = langfuse.get_langfuse_handler()
    assert isinstance(handler, DummyCallbackHandler)
    assert "langfuse_client" in handler.kwargs or "client" in handler.kwargs

    context = langfuse.create_span(name="test")
    assert context == ("span", {"name": "test"})
