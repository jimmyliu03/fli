"""Tests for ``Client`` session warmup.

The warmup GET primes the session with Google's ``NID`` cookie so the
subsequent shopping POST doesn't hit Google's empty-payload fallback for
unauthenticated callers. Tests mock the HTTP layer — they do not call the
real Google Flights endpoint.
"""

from unittest.mock import MagicMock, patch

from fli.search.client import Client


class TestClientWarmup:
    """The session warmup GET runs on client construction."""

    def test_warmup_issues_get_to_google_flights(self):
        """``Client()`` issues exactly one GET to ``WARMUP_URL``."""
        with patch("fli.search.client.requests.Session") as session_cls:
            session = MagicMock()
            session_cls.return_value = session
            Client()
        session.get.assert_called_once()
        call_args = session.get.call_args
        assert call_args.args[0] == Client.WARMUP_URL
        assert call_args.kwargs.get("impersonate") == "chrome"
        assert call_args.kwargs.get("allow_redirects") is True
        assert call_args.kwargs.get("timeout") == Client.WARMUP_TIMEOUT_SECONDS

    def test_warmup_failure_is_swallowed(self):
        """Construction succeeds even when the warmup GET raises."""
        with patch("fli.search.client.requests.Session") as session_cls:
            session = MagicMock()
            session.get.side_effect = RuntimeError("network down")
            session_cls.return_value = session
            # Must not raise.
            client = Client()
        assert client._client is session
        session.get.assert_called_once()

    def test_headers_applied_before_warmup(self):
        """Default headers are set on the session before the warmup GET so
        the cookie jar is populated under the same UA hint the POST uses."""
        with patch("fli.search.client.requests.Session") as session_cls:
            session = MagicMock()
            call_order: list[str] = []
            session.headers.update.side_effect = lambda *_: call_order.append(
                "headers"
            )
            session.get.side_effect = lambda *_a, **_kw: call_order.append("get")
            session_cls.return_value = session
            Client()
        assert call_order == ["headers", "get"]
