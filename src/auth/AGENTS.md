# Auth Module (src/auth)

The auth module implements the **authentication middleware and utilities** for the Agent Platform. It is focused on verifying JSON Web Tokens (JWTs) issued by Azure Entra ID and ensuring that only authorized users can access the agent service endpoints. This module is crucial for securing the API, especially when deployed in environments with multiple users or external access.

## Purpose

The purpose of the auth module is to provide:
- A reusable ASGI middleware (`AuthMiddleware`) that enforces authentication on incoming requests.
- A JWT validation function (`verify_jwt`) that can be used as a dependency in FastAPI routes if needed (though in our design the middleware is primarily used).
- Helper functions to obtain necessary data for token verification (like the JSON Web Key set (JWKS) from Azure, expected issuer and audience strings).

By centralizing these in `src/auth`, we separate security concerns from business logic, making the system easier to maintain and audit.

## How Authentication Works

**Azure AD Token Verification:** The system is designed to accept **OAuth2 access tokens** (JWTs) from Azure AD. The high-level process:
1. A client must acquire an access token from Azure AD (for the configured app) via the appropriate OAuth2 flow.
2. The client sends requests to the agent service with an `Authorization: Bearer <token>` header.
3. The AuthMiddleware intercepts each request:
   - If `AUTH_ENABLED` is False (as set in settings for dev/test), it bypasses checks and simply inserts a dummy user (for development convenience).
   - If auth is enabled, it checks if the path is in a **whitelist** of unauthenticated endpoints (like health checks or static content). By default, it whitelists a few specific paths (e.g., `/chat/playground`, `/favicon.ico`) which might be used for a UI or playground that doesn’t require login.
   - It allows CORS preflight (OPTIONS requests) to pass through without auth to not break browser behavior.
   - For other requests, it expects an `Authorization` header. If missing or not a Bearer token, it immediately returns a 401 Unauthorized with `{"detail": "Missing bearer token"}`.
   - If a token is present, it calls `decode_bearer(token, audience, issuer, jwks_uri)` to validate it.
     - On success, `decoded` will be the JWT claims (likely including user ID, email, etc.). The middleware then stores `decoded` in `request.state.user` and allows the request to proceed to the main app.
     - On failure (token expired, invalid signature, wrong audience, etc.), it returns 401 with `{"detail": "Invalid token"}`.
4. In the FastAPI endpoint, we can then trust that `request.state.user` contains the user’s identity claims (or in dev mode, a dummy value). Endpoints like file upload check for `request.state.user` and will not proceed if it's missing, thus ensuring the middleware did its job.

**Token Validation Details:** The `auth/jwt.py` provides the `decode_bearer` function (imported as `decode_bearer` in middleware). It likely does:
- Use a cached JWKS client for Azure (PyJWKClient from `pyjwt`) to fetch the signing keys.
- Use PyJWT’s `jwt.decode(..., audience=..., issuer=..., algorithms=["RS256"])` to verify the token signature and claims.
- The expected `issuer` is constructed from the Azure AD tenant ID (if not configured explicitly). In code, `get_expected_issuer()` returns something like `https://login.microsoftonline.us/{TENANT_ID}/v2.0`  (note: for Azure Government cloud `.us` domain; in public Azure it would be `.com`).
- The expected `audience` is `api://<AZURE_AD_API_CLIENT_ID>`. This means the Azure AD application representing our API should have that audience URI. If tokens have a different audience (like client ID), validation will fail, which prevents tokens meant for other services from being used.
- If the token is expired (`ExpiredSignatureError`) or otherwise invalid (`InvalidTokenError`), the function will throw, which our middleware catches and responds accordingly.

The `verify_jwt` function in `auth.py` is a FastAPI dependency that does similar checks:
- It’s not heavily used in our code because we rely on middleware, but it’s defined as an alternative or for use in specific routers. It calls similar logic: if no token, raise 401; if auth disabled, skip; otherwise decode and raise exceptions on error.
- Since we went with middleware, `verify_jwt` might be more of a legacy approach or for optional double validation.

**Caching and Performance:** The middleware fetches OIDC configuration (which includes the JWKS URI) once on startup (synchronously, via `requests.get(...)`). It caches the JWKS client (`get_jwk_client()` in auth.py uses lru_cache). So, token verification is fast after the first few calls. The RSA public keys are reused without refetching unless the app restarts (or until Azure rotates keys, which is infrequent and would require cache invalidation or app restart).


## How Auth Module Interacts with Others

- The service adds `AuthMiddleware`. The service also uses functions from `auth.py` for verifying JWT in specific cases (not currently needed due to middleware).
- The `files_router` and potentially other parts use `request.state.user` to derive user IDs for data segregation.
- The core settings define `WHITELIST` (list of paths to skip auth) and the Azure config values. Auth module reads from `settings` inside middleware (it’s passed in).
- During development, if running without Docker, one might disable auth or supply fake tokens. The platform doesn’t currently have a dev token issuance; for local testing with auth on, you’d need to obtain a token from an Azure AD app or mock the decode function.
