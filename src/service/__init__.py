import truststore
truststore.inject_into_ssl()

# Register Entra ID JWT auth middleware (ensure `service/auth.py` initializes it)
import service.auth  # noqa: F401

from service.service import app

__all__ = ["app"]
