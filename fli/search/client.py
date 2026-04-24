"""HTTP client implementation with impersonation, rate limiting and retry functionality.

This module provides a robust HTTP client that handles:
- User agent impersonation (to mimic a browser)
- Rate limiting (10 requests per second)
- Automatic retries with exponential backoff
- Session management
- Error handling
"""

from typing import Any

from curl_cffi import requests
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

client = None


class Client:
    """HTTP client with built-in rate limiting, retry and user agent impersonation functionality."""

    DEFAULT_HEADERS = {
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    }
    WARMUP_URL = "https://www.google.com/travel/flights"
    WARMUP_TIMEOUT_SECONDS = 10

    def __init__(self, proxy: str | None = None):
        """Initialize a new client session with default headers.

        Issues a single GET to ``WARMUP_URL`` so the session picks up Google's
        ``NID`` cookie before any shopping POST. Without this, Google's
        ``GetShoppingResults`` endpoint intermittently serves an empty
        payload for unauthenticated callers — particularly on cold
        serverless workers where the session has no cookie state. Failure
        to warm up is non-fatal; the real POST will still be attempted.

        Args:
            proxy: Optional HTTP proxy URL (e.g. ``http://user:pass@host:port``).
                When set, both HTTP and HTTPS traffic — including the warmup
                GET — routes through the proxy. Pass ``None`` for a direct
                connection.

        """
        self.proxy = proxy
        self._client = requests.Session()
        if proxy:
            self._client.proxies = {"http": proxy, "https": proxy}
        self._client.headers.update(self.DEFAULT_HEADERS)
        self.warm_up_session()

    def warm_up_session(self) -> bool:
        """Prime the session with Google's standard browser cookies.

        Called automatically during ``__init__``; callers can also invoke
        it manually to refresh cookies mid-session (e.g. after an empty
        ``GetShoppingResults`` payload that suggests the cookie jar
        expired). Any error (DNS, TLS, non-2xx, timeout) is swallowed.

        Returns:
            ``True`` if the GET completed with a 2xx response, ``False``
            otherwise. Callers can log the result for observability but
            should not treat a failure as fatal — the subsequent POST
            will still be attempted.
        """
        try:
            resp = self._client.get(
                self.WARMUP_URL,
                impersonate="chrome",
                allow_redirects=True,
                timeout=self.WARMUP_TIMEOUT_SECONDS,
            )
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    def __del__(self):
        """Clean up client session on deletion."""
        if hasattr(self, "_client"):
            self._client.close()

    @sleep_and_retry
    @limits(calls=10, period=1)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a rate-limited GET request with automatic retries.

        Args:
            url: Target URL for the request
            **kwargs: Additional arguments passed to requests.get()

        Returns:
            Response object from the server

        Raises:
            Exception: If request fails after all retries

        """
        try:
            response = self._client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"GET request failed: {str(e)}") from e

    @sleep_and_retry
    @limits(calls=10, period=1)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a rate-limited POST request with automatic retries.

        Args:
            url: Target URL for the request
            **kwargs: Additional arguments passed to requests.post()

        Returns:
            Response object from the server

        Raises:
            Exception: If request fails after all retries

        """
        try:
            response = self._client.post(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"POST request failed: {str(e)}") from e


def get_client(proxy: str | None = None) -> Client:
    """Get or create a shared HTTP client instance.

    The module caches a single client instance. When ``proxy`` differs from
    the cached instance's proxy (including switching to or from ``None``),
    the cache is rebuilt so a subsequent request exits through the new
    transport. Rebuilding re-runs the warmup GET, which is cheap and
    necessary — Google's NID cookie is tied to the exit IP.

    Args:
        proxy: Optional HTTP proxy URL. See :class:`Client` for format.

    Returns:
        Singleton instance of the HTTP client, configured for ``proxy``.

    """
    global client
    if client is None or client.proxy != proxy:
        client = Client(proxy=proxy)
    return client
