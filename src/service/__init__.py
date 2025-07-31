
from service.service import app
import truststore
truststore.inject_into_ssl()

__all__ = ["app"]

