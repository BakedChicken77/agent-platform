
import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    defaults = {
        "OPENAI_API_KEY": "test-openai-key",
        "LANGFUSE_PUBLIC_KEY": "test-langfuse-public",
        "LANGFUSE_SECRET_KEY": "test-langfuse-secret",
        "AZURE_AD_TENANT_ID": "test-tenant",
        "AZURE_AD_CLIENT_ID": "test-client",
        "AZURE_AD_CLIENT_SECRET": "test-secret",
        "AZURE_AD_API_CLIENT_ID": "test-api-client",
        "STREAMLIT_REDIRECT_URI": "http://localhost/callback",
    }
    with patch.dict(os.environ, defaults, clear=True):
        yield

