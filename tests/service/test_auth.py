
from unittest.mock import patch

from auth.jwt import TokenValidationError


def test_auth_disabled_allows_requests(mock_settings, mock_agent, test_client):
    """Auth disabled should allow requests with or without a token."""
    mock_settings.AUTH_ENABLED = False
    mock_settings.refresh_auth_middleware()
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer any-token"},
    )
    assert response.status_code == 200

    # Should also work without any auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 200


def test_auth_enabled_valid_token(mock_settings, mock_agent, test_client):
    """When auth is enabled a valid bearer token should be accepted."""
    mock_settings.AUTH_ENABLED = True
    with patch("requests.get") as mock_get, patch(
        "auth.middleware.decode_bearer", return_value={"sub": "user"}
    ):
        mock_get.return_value.json.return_value = {
            "issuer": "https://login.microsoftonline.us/test-tenant/v2.0",
            "jwks_uri": "https://example.com/.well-known/jwks.json",
        }
        mock_settings.refresh_auth_middleware()
        response = test_client.post(
            "/invoke",
            json={"message": "test"},
            headers={"Authorization": "Bearer good-token"},
        )
    assert response.status_code == 200


def test_auth_enabled_invalid_token(mock_settings, mock_agent, test_client):
    """Invalid or missing bearer tokens should be rejected when auth is enabled."""
    mock_settings.AUTH_ENABLED = True
    with patch("requests.get") as mock_get, patch(
        "auth.middleware.decode_bearer",
        side_effect=TokenValidationError("bad token"),
    ):
        mock_get.return_value.json.return_value = {
            "issuer": "https://login.microsoftonline.us/test-tenant/v2.0",
            "jwks_uri": "https://example.com/.well-known/jwks.json",
        }
        mock_settings.refresh_auth_middleware()
        response = test_client.post(
            "/invoke",
            json={"message": "test"},
            headers={"Authorization": "Bearer wrong-secret"},
        )
        assert response.status_code == 401

        # Should also reject requests with no auth header
        response = test_client.post("/invoke", json={"message": "test"})
        assert response.status_code == 401

