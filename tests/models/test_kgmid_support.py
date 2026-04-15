"""Tests for Knowledge Graph id (kgmid) location support in FlightSegment.

kgmid strings like ``/m/02_286`` (New York City) represent city-level searches
that span all airports in the city, rather than a single IATA-coded airport.
"""

import json
from datetime import datetime, timedelta

import pytest

from fli.models import (
    Airport,
    FlightSearchFilters,
    FlightSegment,
    PassengerInfo,
    TripType,
)


@pytest.fixture
def future_date() -> str:
    """Date 30 days in the future as YYYY-MM-DD."""
    return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")


def test_flight_segment_accepts_kgmid_departure(future_date: str) -> None:
    segment = FlightSegment(
        departure_airport=[["/m/02_286", 0]],
        arrival_airport=[[Airport.LAX, 0]],
        travel_date=future_date,
    )
    assert segment.departure_airport == [["/m/02_286", 0]]


def test_flight_segment_accepts_kgmid_arrival(future_date: str) -> None:
    segment = FlightSegment(
        departure_airport=[[Airport.LAX, 0]],
        arrival_airport=[["/m/04jpl", 0]],
        travel_date=future_date,
    )
    assert segment.arrival_airport == [["/m/04jpl", 0]]


def test_flight_segment_accepts_both_kgmid(future_date: str) -> None:
    segment = FlightSegment(
        departure_airport=[["/m/02_286", 0]],
        arrival_airport=[["/m/04jpl", 0]],
        travel_date=future_date,
    )
    assert segment.departure_airport == [["/m/02_286", 0]]
    assert segment.arrival_airport == [["/m/04jpl", 0]]


def test_flight_segment_rejects_iata_string(future_date: str) -> None:
    with pytest.raises(ValueError, match="kgmid"):
        FlightSegment(
            departure_airport=[["JFK", 0]],
            arrival_airport=[[Airport.LAX, 0]],
            travel_date=future_date,
        )


def test_flight_segment_rejects_malformed_kgmid(future_date: str) -> None:
    with pytest.raises(ValueError, match="kgmid"):
        FlightSegment(
            departure_airport=[["/foo/bar", 0]],
            arrival_airport=[[Airport.LAX, 0]],
            travel_date=future_date,
        )


def test_flight_segment_rejects_bad_slot_type(future_date: str) -> None:
    with pytest.raises(ValueError, match="Slot index"):
        FlightSegment(
            departure_airport=[["/m/02_286", "0"]],
            arrival_airport=[[Airport.LAX, 0]],
            travel_date=future_date,
        )


def test_flight_segment_rejects_wrong_entry_shape(future_date: str) -> None:
    with pytest.raises(ValueError, match="Airport entry"):
        FlightSegment(
            departure_airport=[["/m/02_286"]],
            arrival_airport=[[Airport.LAX, 0]],
            travel_date=future_date,
        )


def test_flight_segment_rejects_same_kgmid(future_date: str) -> None:
    with pytest.raises(ValueError, match="airports must be different"):
        FlightSegment(
            departure_airport=[["/m/02_286", 0]],
            arrival_airport=[["/m/02_286", 0]],
            travel_date=future_date,
        )


def test_flight_segment_mixed_kinds_not_equal(future_date: str) -> None:
    """Mixed Airport enum vs kgmid string are considered different."""
    segment = FlightSegment(
        departure_airport=[["/m/02_286", 0]],
        arrival_airport=[[Airport.JFK, 0]],
        travel_date=future_date,
    )
    assert segment.departure_airport == [["/m/02_286", 0]]


def test_encode_preserves_kgmid_string(future_date: str) -> None:
    """Serializer must pass kgmid string through unchanged to the API payload."""
    filters = FlightSearchFilters(
        trip_type=TripType.ONE_WAY,
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[["/m/02_286", 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=future_date,
            )
        ],
    )
    formatted = filters.format()
    segment_block = formatted[1][13][0]
    assert segment_block[0] == [[["/m/02_286", 0]]]
    assert segment_block[1] == [[["LAX", 0]]]


def test_encoded_url_contains_kgmid(future_date: str) -> None:
    """URL-encoded payload contains the kgmid literally, not an enum name."""
    filters = FlightSearchFilters(
        trip_type=TripType.ONE_WAY,
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[["/m/02_286", 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=future_date,
            )
        ],
    )
    encoded = filters.encode()
    # urllib.parse.quote leaves '/' unescaped by default, so kgmid appears literally.
    assert "/m/02_286" in encoded


def test_resolve_location_iata() -> None:
    from fli.core.parsers import resolve_location

    assert resolve_location("JFK") is Airport.JFK
    assert resolve_location("jfk") is Airport.JFK


def test_resolve_location_kgmid() -> None:
    from fli.core.parsers import resolve_location

    assert resolve_location("/m/02_286") == "/m/02_286"
    assert resolve_location("/m/04jpl") == "/m/04jpl"


def test_resolve_location_invalid() -> None:
    from fli.core.parsers import ParseError, resolve_location

    with pytest.raises(ParseError):
        resolve_location("NOT_AN_AIRPORT")
    with pytest.raises(ParseError):
        resolve_location("/foo/bar")


def test_multiple_airports_per_list_all_kgmid(future_date: str) -> None:
    """When passing multiple kgmid ids in one endpoint list, all must validate."""
    segment = FlightSegment(
        departure_airport=[["/m/02_286", 0], ["/m/04jpl", 0]],
        arrival_airport=[[Airport.LAX, 0]],
        travel_date=future_date,
    )
    assert len(segment.departure_airport) == 2


def test_location_entry_iata() -> None:
    from fli.core import AIRPORT_SLOT, location_entry

    assert location_entry("JFK") == [Airport.JFK, AIRPORT_SLOT]
    assert location_entry("jfk") == [Airport.JFK, AIRPORT_SLOT]
    assert location_entry(Airport.LAX) == [Airport.LAX, AIRPORT_SLOT]


def test_location_entry_kgmid() -> None:
    from fli.core import KGMID_SLOT, location_entry

    assert location_entry("/m/02_286") == ["/m/02_286", KGMID_SLOT]
    assert location_entry("/m/04jpl") == ["/m/04jpl", KGMID_SLOT]


def test_location_entry_invalid() -> None:
    from fli.core import location_entry
    from fli.core.parsers import ParseError

    with pytest.raises(ParseError):
        location_entry("XYZBAD")
    with pytest.raises(ParseError):
        location_entry("/not/a/kgmid")


def test_location_entry_builds_valid_segment(future_date: str) -> None:
    """location_entry output can be used directly in FlightSegment."""
    from fli.core import location_entry

    segment = FlightSegment(
        departure_airport=[location_entry("/m/02_286")],
        arrival_airport=[location_entry("LAX")],
        travel_date=future_date,
    )
    assert segment.departure_airport[0][0] == "/m/02_286"
    assert segment.departure_airport[0][1] == 5
    assert segment.arrival_airport[0][0] is Airport.LAX
    assert segment.arrival_airport[0][1] == 0


def test_encode_roundtrip_with_kgmid(future_date: str) -> None:
    """Round-trip segments with kgmid on both legs encode without error."""
    return_date = (
        datetime.strptime(future_date, "%Y-%m-%d") + timedelta(days=7)
    ).strftime("%Y-%m-%d")
    filters = FlightSearchFilters(
        trip_type=TripType.ROUND_TRIP,
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[["/m/02_286", 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=future_date,
            ),
            FlightSegment(
                departure_airport=[[Airport.LAX, 0]],
                arrival_airport=[["/m/02_286", 0]],
                travel_date=return_date,
            ),
        ],
    )
    formatted = filters.format()
    # The encoded segments list should have two entries
    segments = formatted[1][13]
    assert len(segments) == 2
    # Both kgmid entries preserved
    assert segments[0][0] == [[["/m/02_286", 0]]]
    assert segments[1][1] == [[["/m/02_286", 0]]]
    # Valid JSON round-trip through the encoder
    json.dumps(formatted)
