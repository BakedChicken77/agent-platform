
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
    with patch.dict(
        os.environ,
        {
            "USE_FAKE_MODEL": "true",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "LANGFUSE_HOST": "https://example.com",
            "AUTH_ENABLED": "false",
            "AZURE_AD_TENANT_ID": "test-tenant",
            "AZURE_AD_API_CLIENT_ID": "test-api-client",
            "DISABLE_FEEDBACK_WIDGET": "true",
        },
        clear=True,
    ):
        yield

