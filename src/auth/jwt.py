"""JWT token verification helpers (Azure AD)."""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import Any, Dict

import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)


class TokenValidationError(RuntimeError):
    """Raised when an incoming JWT cannot be validated."""


@cache
def _jwks_client(jwks_uri: str) -> PyJWKClient:  # pragma: no cover
    return PyJWKClient(jwks_uri)


def decode_bearer(token: str, audience: str, issuer: str, jwks_uri: str) -> Dict[str, Any]:
    """Validate *token* and return decoded claims.

    Args:
        token: Raw ``Bearer …`` string without the prefix.
        audience: Expected *aud* claim (API Application ID URI or raw client ID).
        issuer:  Expected *iss* claim (Azure AD tenant issuer).
        jwks_uri: JWKS endpoint for the tenant.

    Raises:
        TokenValidationError: On expiry or signature / claim mismatch.

    Returns:
        Decoded JWT claims as dictionary.
    """

    # FIX 3: Ensure correct scope/resource was requested when acquiring the token,
    # e.g., scope="api://<your-client-id>/.default"

    client = _jwks_client(jwks_uri)

    # FIX 1: Inspect and log actual token audience for debugging
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
        actual_aud = unverified.get("aud")
        logger.info("Token audience claim: %s", actual_aud)
    except jwt.InvalidTokenError:
        logger.info("Failed to decode token without verification", exc_info=True)

    try:
        # First attempt: strict audience check
        signing_key = client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],  # RS256
            audience=audience,
            issuer=issuer,
            options={"verify_aud": True, "verify_iss": True},
        )
    except jwt.InvalidAudienceError as exc:
        # FIX 2: Fallback to accept either client ID or App ID URI as audience
        client_id = os.getenv("AZURE_CLIENT_ID")
        fallback_audiences = [audience]
        if client_id:
            # Accept raw client ID
            if client_id not in fallback_audiences:
                fallback_audiences.append(client_id)
            # Accept App ID URI format
            uri_aud = f"api://{client_id}"
            if uri_aud not in fallback_audiences:
                fallback_audiences.append(uri_aud)
        logger.info("Retrying decode with fallback audiences: %s", fallback_audiences)
        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=fallback_audiences,
                issuer=issuer,
                options={"verify_aud": True, "verify_iss": True},
            )
        except jwt.InvalidTokenError as exc2:
            logger.info("Invalid JWT on fallback", exc_info=exc2)
            raise TokenValidationError(str(exc2)) from exc2
    except jwt.ExpiredSignatureError as exc:
        logger.info("JWT expired", exc_info=exc)
        raise TokenValidationError("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        logger.info("Invalid JWT", exc_info=exc)
        raise TokenValidationError(str(exc)) from exc
