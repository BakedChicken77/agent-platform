# src/service/auth.py

import logging
from functools import lru_cache
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, ExpiredSignatureError, InvalidTokenError

from core import settings

logger = logging.getLogger(__name__)

# Define reusable Bearer token scheme
bearer_scheme = HTTPBearer(auto_error=False)

@lru_cache()
def get_jwk_client() -> PyJWKClient:
    """Returns a cached PyJWKClient for verifying Microsoft Entra ID tokens."""
    if not settings.JWKS_URL:
        raise RuntimeError("Missing JWKS URL in settings.")
    return PyJWKClient(settings.JWKS_URL)

def get_expected_audience() -> str:
    """Returns the expected audience value for JWTs."""
    aud = settings.AZURE_AD_API_CLIENT_ID
    if not aud:
        raise RuntimeError("AZURE_AD_API_CLIENT_ID is not configured.")
    return f"api://{aud}"

def get_expected_issuer() -> str:
    """Returns the expected issuer value for JWTs."""
    if not settings.AZURE_AD_TENANT_ID:
        raise RuntimeError("AZURE_AD_TENANT_ID is not configured.")
    return f"https://login.microsoftonline.us/{settings.AZURE_AD_TENANT_ID}/v2.0"

async def verify_jwt(
    http_auth: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]
) -> None:
    """FastAPI dependency to validate JWT access tokens via Azure AD JWKS."""
    if not settings.AUTH_ENABLED:
        return  # Auth is disabled (e.g., dev/test mode)

    if not http_auth or not http_auth.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    token = http_auth.credentials
    jwk_client = get_jwk_client()

    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token).key
        jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=get_expected_audience(),
            issuer=get_expected_issuer(),
        )
    except ExpiredSignatureError:
        logger.warning("JWT token has expired.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception as e:
        logger.error(f"Unexpected JWT error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")
