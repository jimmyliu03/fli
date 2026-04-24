"""Flight search implementation.

This module provides the core flight search functionality, interfacing directly
with Google Flights' API to find available flights and their details.
"""

import json
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

from fli.core import extract_currency_from_price_token
from fli.models import (
    Airline,
    Airport,
    FlightLeg,
    FlightResult,
    FlightSearchFilters,
)
from fli.models.google_flights.base import TripType
from fli.search.client import get_client

# Google's GetShoppingResults occasionally serves an empty inner payload for
# unauthenticated callers whose session cookies are missing/expired. When that
# happens, we re-warm the session and retry up to this many total attempts
# before giving up.
_MAX_EMPTY_RESPONSE_RETRIES = 3

# How many characters of the raw response body to include in empty-payload
# diagnostic logs. Enough to reveal HTML/captcha/redirect pages but not so
# long that serverless logs get flooded.
_RAW_BODY_PREVIEW_CHARS = 600


def _describe_json_shape(node: Any, _depth: int = 0, _max_depth: int = 3) -> str:
    """Describe a parsed-JSON structure concisely for diagnostics.

    Lists are rendered as ``list[N](<shape_of_first>, <shape_of_second>, ...)``
    up to the first 3 items; dicts as ``dict{keys=[...]}``. Strings and
    scalars show their type only. Depth is capped so deeply nested
    payloads don't blow up log lines.
    """
    if _depth >= _max_depth:
        return type(node).__name__
    if node is None:
        return "null"
    if isinstance(node, bool):
        return "bool"
    if isinstance(node, (int, float)):
        return "number"
    if isinstance(node, str):
        return f"str(len={len(node)})"
    if isinstance(node, list):
        if not node:
            return "list[0]"
        head = ", ".join(
            _describe_json_shape(item, _depth + 1, _max_depth) for item in node[:3]
        )
        suffix = ", ..." if len(node) > 3 else ""
        return f"list[{len(node)}]({head}{suffix})"
    if isinstance(node, dict):
        keys = list(node.keys())[:5]
        suffix = ", ..." if len(node) > 5 else ""
        return f"dict{{keys={keys}{suffix}}}"
    return type(node).__name__


def _extract_proto_type_marker(body: str) -> str | None:
    """Return the first ``type.googleapis.com/...`` protobuf type URL found.

    Google's batch-execute errors carry the protobuf type URL as a plain
    string inside the body, e.g.
    ``type.googleapis.com/travel.frontend.flights.ErrorResponse``. This
    helper extracts the full type URL (up to the next quote or closing
    bracket) so operators can see instantly whether the response is an
    ErrorResponse, a different protobuf shape, or an HTML page without
    any protobuf marker at all. Returns ``None`` when no marker is present.
    """
    marker = "type.googleapis.com/"
    idx = body.find(marker)
    if idx < 0:
        return None
    end = idx + len(marker)
    # Type URLs terminate on quote, bracket, comma, whitespace, or backslash.
    terminators = set('"\',]} \t\n\\')
    while end < len(body) and body[end] not in terminators:
        end += 1
    return body[idx:end]


# Protobuf type URLs we treat as deterministic shopping-endpoint failures —
# Google will return the same error no matter how many times we retry with
# re-warmed cookies, so the shopping POST should fail fast and let the
# caller fall back to a different query shape.
_DETERMINISTIC_ERROR_PROTO_TYPES = frozenset(
    {
        "type.googleapis.com/travel.frontend.flights.ErrorResponse",
    }
)


def _is_deterministic_error_response(body: str) -> bool:
    """Return True when the response body embeds a known-deterministic error.

    Checked via substring against ``_DETERMINISTIC_ERROR_PROTO_TYPES`` rather
    than full JSON parsing so we can flag malformed or only-partially-JSON
    responses too.
    """
    proto_type = _extract_proto_type_marker(body)
    return proto_type in _DETERMINISTIC_ERROR_PROTO_TYPES


class SearchFlights:
    """Flight search implementation using Google Flights' API.

    This class handles searching for specific flights with detailed filters,
    parsing the results into structured data models.
    """

    BASE_URL = "https://www.google.com/_/FlightsFrontendUi/data/travel.frontend.flights.FlightsFrontendService/GetShoppingResults"
    DEFAULT_HEADERS = {
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    }

    def __init__(self, proxy: str | None = None):
        """Initialize the search client for flight searches.

        Args:
            proxy: Optional HTTP proxy URL forwarded to the shared HTTP client.
                See :class:`fli.search.client.Client` for format and semantics.

        """
        self.client = get_client(proxy=proxy)

    def search(
        self,
        filters: FlightSearchFilters,
        top_n: int = 5,
        return_combined_only: bool = False,
    ) -> list[FlightResult | tuple[FlightResult, ...]] | None:
        """Search for flights using the given FlightSearchFilters.

        Args:
            filters: Full flight search object including airports, dates, and preferences
            top_n: Number of flights to limit the return flight search to
            return_combined_only: If True, for round-trip/multi-city searches with no
                ``selected_flight`` on any segment, return the initial outbound list
                without recursing into per-outbound return fetches. Each outbound's
                ``price`` is Google's cheapest-combined round-trip price (matches the
                Google Flights UI outbound list). Default False preserves the original
                recursive behavior that yields ``(outbound, return)`` tuples.

        Returns:
            List of FlightResult objects (one-way, or round-trip outbound list when
            ``return_combined_only`` is True), tuples of FlightResult (round-trip or
            multi-city), or None if no results

        Raises:
            Exception: If the search fails or returns invalid data

        Note:
            Multi-city searches (TripType.MULTI_CITY) with distinct city pairs may
            time out due to limitations of the Google Flights API endpoint.  The
            endpoint reliably supports one-way and round-trip searches.

        """
        encoded_filters = filters.encode()

        try:
            parsed = self._post_and_extract_payload(encoded_filters)
            if not parsed:
                return None

            encoded_filters = json.loads(parsed)
            flights_data = [
                item
                for i in [2, 3]
                if isinstance(encoded_filters[i], list)
                for item in encoded_filters[i][0]
            ]
            flights = [self._parse_flights_data(flight) for flight in flights_data]

            if filters.trip_type == TripType.ONE_WAY:
                return flights

            # For round-trip and multi-city, iteratively select each leg
            # and fetch the next leg's options with combined pricing.
            num_segments = len(filters.flight_segments)
            selected_count = sum(
                1 for s in filters.flight_segments if s.selected_flight is not None
            )

            # If all previous segments are selected, we're on the last leg
            if selected_count >= num_segments - 1:
                return flights

            # Caller wants the initial outbound list with Google's cheapest-combined
            # round-trip prices, without recursing into per-outbound return fetches.
            if return_combined_only and selected_count == 0:
                return flights

            # Select each flight option and fetch the next leg
            flight_combos = []
            for selected_flight in flights[:top_n]:
                next_filters = deepcopy(filters)
                next_filters.flight_segments[selected_count].selected_flight = selected_flight
                next_results = self.search(next_filters, top_n=top_n)
                if next_results is not None:
                    for next_result in next_results:
                        if isinstance(next_result, tuple):
                            flight_combos.append((selected_flight,) + next_result)
                        else:
                            flight_combos.append((selected_flight, next_result))

            return flight_combos

        except Exception as e:
            raise Exception(f"Search failed: {str(e)}") from e

    def _post_and_extract_payload(self, encoded_filters: str) -> str | None:
        """POST the shopping request and return the inner payload string.

        Google's ``GetShoppingResults`` occasionally returns ``null`` at
        ``response[0][2]`` even after a successful 2xx — typically when
        session cookies are missing/expired. Those cases are transient
        and recover after a re-warm, so we retry up to
        ``_MAX_EMPTY_RESPONSE_RETRIES`` total attempts with a re-warmed
        session between attempts.

        However, some queries produce a *deterministic* empty response —
        Google embeds a ``travel.frontend.flights.ErrorResponse`` protobuf
        in the body (gRPC code 13 = INTERNAL) and will return the same
        error on every subsequent request. Retrying wastes ~22s per
        attempt and blocks the caller's fallback. In that case we fail
        fast after the first attempt.

        On each empty-payload attempt a diagnostic line is logged to
        stderr: status, body size, extracted protobuf type (if any),
        outer JSON shape, and a capped body prefix.

        Returns the non-empty inner payload string on success, or
        ``None`` on an empty/deterministic-error response or after all
        transient retries are exhausted.
        """
        last_parsed: str | None = None
        for attempt in range(1, _MAX_EMPTY_RESPONSE_RETRIES + 1):
            response = self.client.post(
                url=self.BASE_URL,
                data=f"f.req={encoded_filters}",
                impersonate="chrome",
                allow_redirects=True,
            )
            response.raise_for_status()
            last_parsed = self._parse_inner_payload(response.text)
            if last_parsed:
                if attempt > 1:
                    print(
                        f"[fli] shopping POST succeeded on attempt {attempt}",
                        file=sys.stderr,
                        flush=True,
                    )
                return last_parsed

            self._log_empty_response(attempt, response)

            # If Google embedded an ErrorResponse protobuf, the failure
            # is deterministic for this filter shape — retrying produces
            # the same error. Fail fast so the caller can fall back.
            if _is_deterministic_error_response(getattr(response, "text", "")):
                print(
                    "[fli] response contains ErrorResponse protobuf — "
                    "failing fast, retry would not help",
                    file=sys.stderr,
                    flush=True,
                )
                return None

            warmed = self.client.warm_up_session()
            print(
                f"[fli] shopping POST returned empty payload on attempt "
                f"{attempt}/{_MAX_EMPTY_RESPONSE_RETRIES} "
                f"(re-warm ok={warmed})",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[fli] giving up after {_MAX_EMPTY_RESPONSE_RETRIES} empty payloads",
            file=sys.stderr,
            flush=True,
        )
        return last_parsed

    @staticmethod
    def _parse_inner_payload(raw_text: str) -> str | None:
        """Extract the inner payload string at ``response[0][2]`` if present.

        Returns ``None`` if the outer JSON doesn't have the expected shape
        (too short, wrong types, etc.). Does not raise — callers
        distinguish between legitimate empty payloads and structural
        malformation via the accompanying raw-response log.
        """
        try:
            outer = json.loads(raw_text.lstrip(")]}'"))
            if isinstance(outer, list) and outer and isinstance(outer[0], list):
                if len(outer[0]) > 2:
                    inner = outer[0][2]
                    return inner if inner else None
            return None
        except (ValueError, TypeError, IndexError):
            return None

    @staticmethod
    def _log_empty_response(attempt: int, response: Any) -> None:
        """Dump diagnostics for an empty-payload response to stderr.

        Captures status, body size, a capped prefix of the body, the outer
        JSON shape (when parseable), and — when the body encodes a
        protobuf ``type.googleapis.com/...`` marker — the protobuf type
        name so operators can tell instantly whether Google returned an
        ``ErrorResponse``, a captcha page, or some other shape. Safe
        against exceptions: any failure inside the diagnostic itself is
        logged rather than propagated.
        """
        try:
            body = getattr(response, "text", "") or ""
            status = getattr(response, "status_code", "?")
            # Strip the XSSI prefix so the outer JSON is parseable. Keep a
            # capped prefix of the RAW body for eyeballing (captchas,
            # redirect pages, etc. never have the XSSI prefix at all).
            trimmed = body.lstrip(")]}'").lstrip("\n")
            body_preview = body[:_RAW_BODY_PREVIEW_CHARS]
            if len(body) > _RAW_BODY_PREVIEW_CHARS:
                body_preview += f"... [+{len(body) - _RAW_BODY_PREVIEW_CHARS} chars]"
            shape_desc: str
            proto_type = _extract_proto_type_marker(body)
            try:
                outer = json.loads(trimmed) if trimmed else None
                shape_desc = _describe_json_shape(outer)
            except ValueError as parse_err:
                shape_desc = f"<not JSON: {parse_err}>"
            print(
                f"[fli] empty-payload diagnostic (attempt {attempt}): "
                f"status={status} body_len={len(body)} "
                f"proto_type={proto_type or '<none>'} "
                f"outer_shape={shape_desc} body_prefix={body_preview!r}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"[fli] could not dump empty-payload diagnostic: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _parse_flights_data(data: list) -> FlightResult:
        """Parse raw flight data into a structured FlightResult.

        Args:
            data: Raw flight data from the API response

        Returns:
            Structured FlightResult object with all flight details

        """
        price, currency = SearchFlights._parse_price_info(data)
        flight = FlightResult(
            price=price,
            currency=currency,
            duration=data[0][9],
            stops=len(data[0][2]) - 1,
            legs=[
                FlightLeg(
                    airline=SearchFlights._parse_airline(fl[22][0]),
                    flight_number=fl[22][1],
                    departure_airport=SearchFlights._parse_airport(fl[3]),
                    arrival_airport=SearchFlights._parse_airport(fl[6]),
                    departure_datetime=SearchFlights._parse_datetime(fl[20], fl[8]),
                    arrival_datetime=SearchFlights._parse_datetime(fl[21], fl[10]),
                    duration=fl[11],
                )
                for fl in data[0][2]
            ],
        )
        return flight

    @staticmethod
    def _parse_price_info(data: list) -> tuple[float, str | None]:
        """Extract the numeric price and returned currency from raw flight data."""
        price_block = SearchFlights._get_price_block(data)
        price = 0.0
        currency = None
        try:
            if price_block and price_block[0]:
                price = float(price_block[0][-1])
        except (IndexError, TypeError):
            pass
        try:
            if price_block and len(price_block) > 1:
                currency = extract_currency_from_price_token(price_block[1])
        except (IndexError, TypeError):
            pass
        return price, currency

    @staticmethod
    def _parse_currency(data: list) -> str | None:
        """Extract the returned currency code from raw flight data."""
        try:
            price_block = SearchFlights._get_price_block(data)
            if price_block and len(price_block) > 1:
                return extract_currency_from_price_token(price_block[1])
        except (IndexError, TypeError):
            pass
        return None

    @staticmethod
    def _get_price_block(data: list) -> list | None:
        """Return the raw price block attached to a flight row."""
        try:
            if len(data) > 1 and isinstance(data[1], list):
                return data[1]
        except TypeError:
            pass
        return None

    @staticmethod
    def _parse_datetime(date_arr: list[int], time_arr: list[int]) -> datetime:
        """Convert date and time arrays to datetime.

        Args:
            date_arr: List of integers [year, month, day]
            time_arr: List of integers [hour, minute]

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If arrays contain only None values

        """
        if not any(x is not None for x in date_arr) or not any(x is not None for x in time_arr):
            raise ValueError("Date and time arrays must contain at least one non-None value")

        return datetime(*(x or 0 for x in date_arr), *(x or 0 for x in time_arr))

    @staticmethod
    def _parse_airline(airline_code: str) -> Airline:
        """Convert airline code to Airline enum.

        Args:
            airline_code: Raw airline code from API

        Returns:
            Corresponding Airline enum value

        """
        if airline_code[0].isdigit():
            airline_code = f"_{airline_code}"
        return getattr(Airline, airline_code)

    @staticmethod
    def _parse_airport(airport_code: str) -> Airport:
        """Convert airport code to Airport enum.

        Args:
            airport_code: Raw airport code from API

        Returns:
            Corresponding Airport enum value

        """
        return getattr(Airport, airport_code)
