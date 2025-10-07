
import os
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

# Register Entra ID JWT auth middleware (ensure `service/auth.py` initializes it)

if os.getenv("AUTH_ENABLED"):
    import service.auth  # noqa: F401

from service.service import app

__all__ = ["app"]
