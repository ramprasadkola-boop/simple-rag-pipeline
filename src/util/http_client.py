"""
Singleton HTTP client with bounded connection pool (1 connection).

This ensures a single point of connection across the entire application,
preventing resource exhaustion and providing centralized configuration.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance = None
_crawl4ai_available = None


class SingletonHTTPClient:
    """Singleton HTTP session with bounded connection pool."""

    def __init__(self, pool_connections=1, pool_maxsize=1):
        self.session = requests.Session()
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize

        # Configure retry strategy - only retry on status codes, not connection errors
        retries = Retry(
            total=1,
            backoff_factor=0.1,
            status_forcelist=(500, 502, 503, 504),
            connect=0,  # Don't retry on connection errors
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


def is_crawl4ai_available(base_url: str = "http://localhost:11235", timeout: int = 2) -> bool:
    """Check if Crawl4AI server is available (cached result)."""
    global _crawl4ai_available
    
    if _crawl4ai_available is not None:
        return _crawl4ai_available
    
    try:
        session = get_http_client().get_session()
        # Try a lightweight HEAD request to health/ping endpoint
        response = session.head(f"{base_url}/", timeout=timeout, allow_redirects=False)
        _crawl4ai_available = response.status_code < 500
        if _crawl4ai_available:
            logger.debug(f"✓ Crawl4AI available at {base_url}")
        else:
            logger.debug(f"✗ Crawl4AI returned status {response.status_code}")
    except Exception as e:
        _crawl4ai_available = False
        logger.debug(f"✗ Crawl4AI unavailable at {base_url}: {type(e).__name__}")
    
    return _crawl4ai_available


def reset_crawl4ai_availability():
    """Reset cached availability status (for testing)."""
    global _crawl4ai_available
    _crawl4ai_available = None


def get_session() -> requests.Session:
    """Convenience function to get the singleton session directly."""
    return get_http_client().get_session()
