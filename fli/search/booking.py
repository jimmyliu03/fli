"""Google Flights booking-page fare option extraction."""

from __future__ import annotations

import json
import re
import urllib.parse
from html import unescape
from typing import Any

from fli.models import FareOption, FareOptionsResult
from fli.search.client import get_client

BOOKING_RESULTS_URL = (
    "https://www.google.com/_/FlightsFrontendUi/data/"
    "travel.frontend.flights.FlightsFrontendService/GetBookingResults"
)


def extract_booking_fare_options(
    booking_url: str,
    *,
    proxy: str | None = None,
    currency: str | None = None,
) -> FareOptionsResult:
    """Fetch a Google Flights booking URL and return fare-card prices.

    Google renders booking fare cards from the ``GetBookingResults`` endpoint,
    not from the normal shopping response. The request payload is derived from
    the booking page's initial app-state constraints.
    """
    return BookingFareExtractor(proxy=proxy).extract_fare_options(
        booking_url,
        currency=currency,
    )


class BookingFareExtractor:
    """Extract fare-family prices from a Google Flights booking page."""

    DEFAULT_HEADERS = {
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
        "origin": "https://www.google.com",
        "referer": "https://www.google.com/travel/flights/booking",
    }

    def __init__(self, proxy: str | None = None):
        """Initialize the shared Google Flights HTTP client."""
        self.client = get_client(proxy=proxy)

    def extract_fare_options(
        self,
        booking_url: str,
        *,
        currency: str | None = None,
    ) -> FareOptionsResult:
        """Return booking-page fare options for ``booking_url``."""
        page_response = self.client.get(
            booking_url,
            impersonate="chrome",
            allow_redirects=True,
        )
        page_response.raise_for_status()

        try:
            constraints = extract_booking_constraints(page_response.text)
        except ValueError as exc:
            return FareOptionsResult(
                available=False,
                reason=f"booking_constraints_unavailable: {exc}",
            )

        request_body = build_booking_results_request_body(constraints)
        booking_response = self.client.post(
            BOOKING_RESULTS_URL,
            data=f"f.req={urllib.parse.quote(request_body)}",
            headers=self.DEFAULT_HEADERS,
            impersonate="chrome",
            allow_redirects=True,
        )
        booking_response.raise_for_status()

        inferred_currency = currency or _currency_from_booking_url(booking_url)
        fare_options = parse_booking_results_fare_options(
            booking_response.text,
            currency=inferred_currency,
        )
        if not fare_options:
            return FareOptionsResult(
                available=False,
                reason="fare_options_not_exposed_in_booking_results_payload",
            )
        return FareOptionsResult(available=True, fare_options=fare_options)


def build_booking_results_request_body(constraints: Any) -> str:
    """Build the ``f.req`` value for ``GetBookingResults``.

    The endpoint expects the same outer wrapper shape as other FlightsFrontendUi
    data services, with the actual request encoded as a JSON string.
    """
    request = [None, constraints]
    return json.dumps(
        [None, json.dumps(request, separators=(",", ":"))],
        separators=(",", ":"),
    )


def extract_booking_constraints(html: str) -> Any:
    """Extract the Google Flights constraints object from booking page HTML."""
    data = _find_p4ikpb_data_p(html)
    if data is None:
        raise ValueError("p4IKPb data-p not found")

    decoded = unescape(data)
    if decoded.startswith("%.@."):
        decoded = decoded[4:]

    try:
        app_state, _ = json.JSONDecoder().raw_decode(decoded)
        return app_state[0][1][1]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("p4IKPb constraints malformed") from exc


def parse_booking_results_fare_options(
    raw_text: str,
    *,
    currency: str | None = None,
) -> list[FareOption]:
    """Parse fare cards from a ``GetBookingResults`` streaming response."""
    options: list[FareOption] = []
    for record in _iter_streaming_wrb_records(raw_text):
        try:
            payload = json.loads(record[2])
        except (IndexError, TypeError, ValueError):
            continue

        option_nodes = _extract_option_nodes(payload)
        for idx, option_node in enumerate(option_nodes):
            fare_option = _parse_option_node(option_node, idx, currency=currency)
            if fare_option is not None:
                options.append(fare_option)

    return _dedupe_fare_options(options)


def _find_p4ikpb_data_p(html: str) -> str | None:
    for match in re.finditer(r"<c-wiz\b[^>]*>", html):
        tag = match.group(0)
        if 'jsrenderer="p4IKPb"' not in tag:
            continue
        data_match = re.search(r'data-p="([^"]+)"', tag)
        if data_match:
            return data_match.group(1)
    return None


def _iter_streaming_wrb_records(raw_text: str) -> list[list[Any]]:
    lines = raw_text.splitlines()
    records: list[list[Any]] = []
    for index, line in enumerate(lines):
        candidate: str | None = None
        if line.isdigit() and index + 1 < len(lines):
            candidate = lines[index + 1]
        elif line.startswith("[["):
            candidate = line
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if isinstance(item, list) and item and item[0] == "wrb.fr":
                records.append(item)
    return records


def _extract_option_nodes(payload: Any) -> list[Any]:
    try:
        nodes = payload[1][0]
    except (IndexError, TypeError):
        return []
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, list)]


def _parse_option_node(
    node: list[Any],
    index: int,
    *,
    currency: str | None,
) -> FareOption | None:
    price = _extract_total_price(node)
    if price is None:
        return None

    brand = _extract_brand(node)
    if not brand:
        return None

    return FareOption(
        brand=brand,
        cabin=_infer_cabin_from_brand(brand),
        basic_economy="basic" in brand.lower(),
        price=price,
        currency=currency,
        bags=_summarize_bags(node),
        refundability="refundable" if "refundable" in brand.lower() else None,
        changeability=None,
        raw_path=f"$[1][0][{index}]",
    )


def _extract_total_price(node: list[Any]) -> float | None:
    try:
        price = node[7][0][1]
    except (IndexError, TypeError):
        return None
    if isinstance(price, int | float):
        return float(price)
    return None


def _extract_brand(node: list[Any]) -> str | None:
    try:
        details_brand = node[21][3]
        if isinstance(details_brand, str) and details_brand.strip():
            return details_brand.strip()
    except (IndexError, TypeError):
        pass

    try:
        fare_brand = node[14][0][0][1][1]
        if isinstance(fare_brand, str) and fare_brand.strip():
            return fare_brand.strip().replace("_", " ").title()
    except (IndexError, TypeError):
        pass

    return None


def _summarize_bags(node: list[Any]) -> str | None:
    try:
        bag_entries = node[18]
    except (IndexError, TypeError):
        return None
    if not isinstance(bag_entries, list):
        return None
    first_checked_bag = bag_entries[0] if bag_entries else None
    if _contains_checked_bag_fee(first_checked_bag):
        return "first_checked_bag_fee"
    if first_checked_bag == [3]:
        return "first_checked_bag_included"
    return None


def _contains_checked_bag_fee(node: Any) -> bool:
    if isinstance(node, list):
        try:
            if node[0] == 2 and isinstance(node[1], list):
                return True
        except IndexError:
            pass
        return any(_contains_checked_bag_fee(item) for item in node)
    return False


def _infer_cabin_from_brand(brand: str) -> str | None:
    lowered = brand.lower()
    if "first" in lowered:
        return "first"
    if "business" in lowered:
        return "business"
    if "premium" in lowered:
        return "premium_economy"
    if "economy" in lowered or "main cabin" in lowered:
        return "economy"
    return None


def _currency_from_booking_url(booking_url: str) -> str | None:
    parsed = urllib.parse.urlparse(booking_url)
    values = urllib.parse.parse_qs(parsed.query).get("curr")
    if not values:
        return None
    return values[0] or None


def _dedupe_fare_options(options: list[FareOption]) -> list[FareOption]:
    deduped: list[FareOption] = []
    seen: set[tuple[str | None, float | None, str | None]] = set()
    for option in options:
        key = (option.brand, option.price, option.currency)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return deduped
