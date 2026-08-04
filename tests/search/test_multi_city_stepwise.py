import json
from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from fli.models import (
    Airline,
    Airport,
    FlightLeg,
    FlightResult,
    FlightSearchFilters,
    FlightSegment,
    PassengerInfo,
)
from fli.models.google_flights.base import TripType
from fli.search import SearchFlights


def result(number: str = "123") -> FlightResult:
    departure = datetime.now() + timedelta(days=30)
    return FlightResult(
        legs=[
            FlightLeg(
                airline=Airline.UA,
                flight_number=number,
                departure_airport=Airport.SFO,
                arrival_airport=Airport.LAX,
                departure_datetime=departure,
                arrival_datetime=departure + timedelta(hours=1),
                duration=60,
            )
        ],
        price=100,
        currency="USD",
        duration=60,
        stops=0,
    )


def multi_city_filters(count: int = 3) -> FlightSearchFilters:
    airports = [Airport.SFO, Airport.LAX, Airport.SEA, Airport.JFK, Airport.BOS, Airport.MIA]
    start = datetime.now().date() + timedelta(days=30)
    return FlightSearchFilters(
        trip_type=TripType.MULTI_CITY,
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[[airports[index], 0]],
                arrival_airport=[[airports[index + 1], 0]],
                travel_date=(start + timedelta(days=index)).isoformat(),
            )
            for index in range(count)
        ],
    )


def test_next_leg_only_does_not_recurse():
    filters = multi_city_filters()
    filters.flight_segments[0].selected_flight = result()
    candidate = result("456")
    payload = json.dumps([None, None, [[[["marker"]]]], None])
    search = SearchFlights()

    with (
        patch.object(search, "_post_and_extract_payload", return_value=payload) as post,
        patch("fli.search.flights._is_itinerary_entry", return_value=True),
        patch.object(search, "_parse_flights_data", return_value=candidate),
    ):
        results = search.search(filters, next_leg_only=True)

    assert results == [candidate]
    post.assert_called_once()


def test_multi_city_payload_matches_live_google_shape():
    filters = multi_city_filters()
    formatted = filters.format()

    assert formatted[1][2] == TripType.MULTI_CITY.value
    assert all(segment[14] == 1 for segment in formatted[1][13])
    assert len(formatted[1]) == 18
    assert formatted[2:] == [0, 0, 0, 1]


def test_multi_city_basic_economy_exclusion_uses_index_28():
    filters = multi_city_filters()
    filters.exclude_basic_economy = True
    formatted = filters.format()

    assert len(formatted[1]) == 29
    assert formatted[1][28] == 1


def test_multi_city_rejects_non_contiguous_selected_prefix():
    filters = multi_city_filters()
    filters.flight_segments[1].selected_flight = result()

    with pytest.raises(ValueError, match="contiguous prefix"):
        SearchFlights().search(filters, next_leg_only=True)


@pytest.mark.parametrize("count", [1, 6])
def test_multi_city_rejects_unsupported_leg_count(count: int):
    filters = multi_city_filters(min(count, 5))
    if count == 6:
        extra = deepcopy(filters.flight_segments[-1])
        extra.departure_airport = [[Airport.MIA, 0]]
        extra.arrival_airport = [[Airport.ATL, 0]]
        filters.flight_segments.append(extra)

    with pytest.raises(ValueError, match="between 2 and 5"):
        SearchFlights().search(filters, next_leg_only=True)


def test_multi_city_rejects_decreasing_dates():
    filters = multi_city_filters()
    filters.flight_segments[1].travel_date = filters.flight_segments[0].travel_date
    filters.flight_segments[0].travel_date = filters.flight_segments[2].travel_date

    with pytest.raises(ValueError, match="nondecreasing"):
        SearchFlights().search(filters, next_leg_only=True)
