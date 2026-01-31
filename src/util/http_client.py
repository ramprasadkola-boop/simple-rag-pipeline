"""
Singleton HTTP client with bounded connection pool (1 connection).

This ensures a single point of connection across the entire application,
preventing resource exhaustion and providing centralized configuration.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

_lock = threading.Lock()
_instance = None


class SingletonHTTPClient:
    """Singleton HTTP session with bounded connection pool."""

    def __init__(self, pool_connections=1, pool_maxsize=1):
        self.session = requests.Session()
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize

        # Configure retry strategy
        retries = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=(500, 502, 503, 504),
        )

        # Create adapter with bounded pool (1 connection max)
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retries,
        )

        # Mount adapters for both http and https
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_session(self):
        """Get the underlying requests.Session object."""
        return self.session

    def close(self):
        """Close the session."""
        self.session.close()


def get_http_client() -> SingletonHTTPClient:
    """Get or create the singleton HTTP client instance."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SingletonHTTPClient(pool_connections=1, pool_maxsize=1)
    return _instance


def get_session() -> requests.Session:
    """Convenience function to get the singleton session directly."""
    return get_http_client().get_session()
