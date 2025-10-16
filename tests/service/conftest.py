
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from service import app
from service import tracing
from auth.middleware import AuthMiddleware


@pytest.fixture
def test_client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
    )
    agent_mock.get_state = Mock()  # Default empty mock for get_state
    with patch("service.service.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_langfuse(mock_settings):
    """Provide a mock Langfuse client so tests can assert tracing behaviour."""

    tracing.reset_tracing_state()
    langfuse_client = Mock()
    trace_client = Mock()
    span_client = Mock()
    langfuse_client.trace.return_value = trace_client
    langfuse_client.span.return_value = span_client

    with patch.object(tracing, "get_shared_client", return_value=langfuse_client):
        original = mock_settings.LANGFUSE_TRACING
        mock_settings.LANGFUSE_TRACING = True
        yield {
            "client": langfuse_client,
            "trace": trace_client,
            "span": span_client,
        }
        mock_settings.LANGFUSE_TRACING = original


@pytest.fixture
def mock_settings(mock_env):
    """Fixture to ensure settings are clean for each test."""
    with ExitStack() as stack:
        service_settings = stack.enter_context(patch("service.service.settings"))
        stack.enter_context(patch("service.tracing.settings", service_settings))

        service_settings.AUTH_ENABLED = False
        service_settings.AZURE_AD_TENANT_ID = "test-tenant"
        service_settings.AZURE_AD_API_CLIENT_ID = "test-api-client"
        service_settings.WHITELIST = set()
        service_settings.LANGFUSE_TRACING = False

        def refresh_auth_middleware() -> None:
            for middleware in app.user_middleware:
                if middleware.cls is AuthMiddleware:
                    middleware.kwargs["settings"] = service_settings
                    break
            app.middleware_stack = app.build_middleware_stack()

        refresh_auth_middleware()
        service_settings.refresh_auth_middleware = refresh_auth_middleware
        yield service_settings

    for middleware in app.user_middleware:
        if middleware.cls is AuthMiddleware:
            middleware.kwargs["settings"] = __import__(
                "service.service", fromlist=["settings"]
            ).settings
            break
    app.middleware_stack = app.build_middleware_stack()


@pytest.fixture
def mock_httpx():
    """Patch httpx.stream and httpx.get to use our test client."""

    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.get(path, **kwargs)

        with patch("httpx.stream", mock_stream):
            with patch("httpx.get", mock_get):
                yield

