"""Tests for Search class."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from tenacity import retry, stop_after_attempt, wait_exponential

from fli.core import build_selected_return_segments
from fli.models import (
    Airline,
    Airport,
    FareOptionsResult,
    FlightLeg,
    FlightResult,
    FlightSearchFilters,
    FlightSegment,
    MaxStops,
    PassengerInfo,
    SeatType,
    SortBy,
)
from fli.models.google_flights.base import TripType
from fli.search import SearchFlights


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def search_with_retry(search: SearchFlights, search_params):
    """Search with retry logic for flaky API responses."""
    results = search.search(search_params)
    if not results:
        raise ValueError("Empty results, retrying...")
    return results


@pytest.fixture
def search():
    """Create a reusable Search instance."""
    return SearchFlights()


@pytest.fixture
def basic_search_params():
    """Create basic search params for testing."""
    today = datetime.now()
    future_date = today + timedelta(days=30)
    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=1,
            children=0,
            infants_in_seat=0,
            infants_on_lap=0,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.PHX, 0]],
                arrival_airport=[[Airport.SFO, 0]],
                travel_date=future_date.strftime("%Y-%m-%d"),
            )
        ],
        stops=MaxStops.NON_STOP,
        seat_type=SeatType.ECONOMY,
        sort_by=SortBy.CHEAPEST,
        show_all_results=False,
    )


@pytest.fixture
def complex_search_params():
    """Create more complex search params for testing."""
    today = datetime.now()
    future_date = today + timedelta(days=60)
    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=2,
            children=1,
            infants_in_seat=0,
            infants_on_lap=1,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.JFK, 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=future_date.strftime("%Y-%m-%d"),
            )
        ],
        stops=MaxStops.ONE_STOP_OR_FEWER,
        seat_type=SeatType.FIRST,
        sort_by=SortBy.TOP_FLIGHTS,
        show_all_results=False,
    )


@pytest.fixture
def round_trip_search_params():
    """Create basic round trip search params for testing."""
    today = datetime.now()
    outbound_date = today + timedelta(days=30)
    return_date = outbound_date + timedelta(days=7)

    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=1,
            children=0,
            infants_in_seat=0,
            infants_on_lap=0,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.SFO, 0]],
                arrival_airport=[[Airport.JFK, 0]],
                travel_date=outbound_date.strftime("%Y-%m-%d"),
            ),
            FlightSegment(
                departure_airport=[[Airport.JFK, 0]],
                arrival_airport=[[Airport.SFO, 0]],
                travel_date=return_date.strftime("%Y-%m-%d"),
            ),
        ],
        stops=MaxStops.NON_STOP,
        seat_type=SeatType.ECONOMY,
        sort_by=SortBy.CHEAPEST,
        trip_type=TripType.ROUND_TRIP,
        show_all_results=False,
    )


@pytest.fixture
def complex_round_trip_params():
    """Create more complex round trip search params for testing."""
    today = datetime.now()
    outbound_date = today + timedelta(days=60)
    return_date = outbound_date + timedelta(days=14)

    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=2,
            children=1,
            infants_in_seat=0,
            infants_on_lap=1,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.LAX, 0]],
                arrival_airport=[[Airport.ORD, 0]],
                travel_date=outbound_date.strftime("%Y-%m-%d"),
            ),
            FlightSegment(
                departure_airport=[[Airport.ORD, 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=return_date.strftime("%Y-%m-%d"),
            ),
        ],
        stops=MaxStops.ONE_STOP_OR_FEWER,
        seat_type=SeatType.BUSINESS,
        sort_by=SortBy.TOP_FLIGHTS,
        trip_type=TripType.ROUND_TRIP,
        show_all_results=False,
    )


@pytest.mark.parametrize(
    "search_params_fixture",
    [
        "basic_search_params",
        "complex_search_params",
    ],
)
def test_search_functionality(search, search_params_fixture, request):
    """Test flight search functionality with different data sets."""
    search_params = request.getfixturevalue(search_params_fixture)
    results = search.search(search_params)
    assert isinstance(results, list)


def test_multiple_searches(search, basic_search_params, complex_search_params):
    """Test performing multiple searches with the same Search instance."""
    # First search
    results1 = search.search(basic_search_params)
    assert isinstance(results1, list)

    # Second search with different data
    results2 = search.search(complex_search_params)
    assert isinstance(results2, list)

    # Third search reusing first search data
    results3 = search.search(basic_search_params)
    assert isinstance(results3, list)


# TODO: These round-trip tests hit the live Google Flights API with multiple
# sequential requests (outbound + return for each result), causing frequent
# timeouts on CI runners. They should be refactored to mock the HTTP client
# instead of making real API calls. See GitHub issue for follow-up.
#
# def test_basic_round_trip_search(search, round_trip_search_params):
# def test_complex_round_trip_search(search, complex_round_trip_params):
# def test_round_trip_with_selected_outbound(search, round_trip_search_params):
# def test_round_trip_result_structure(search, search_params_fixture, request):


class TestReturnCombinedOnly:
    """Tests for the ``return_combined_only`` flag on round-trip searches."""

    @staticmethod
    def _flight_result(price: float):
        """Build a minimal FlightResult stand-in for parser output."""
        return MagicMock(price=price)

    @staticmethod
    def _patched_search(flights: list):
        """Return a context manager for the initial HTTP/parse pipeline.

        Mocks ``client.post``, the two ``json.loads`` calls, and
        ``_parse_flights_data`` so the test never hits the network. The second
        ``json.loads`` return pads to length 4 so the ``[2, 3]`` index scan
        finds exactly one list (index 2) and skips index 3.
        """
        search = SearchFlights()
        response = MagicMock()
        response.text = ")]}'\n[]"
        # Each marker must shape-discriminate as an itinerary entry — el[0]
        # is a list — so it survives the non-itinerary filter introduced
        # alongside TravelWarning collection. The inner content doesn't
        # matter because _parse_flights_data is patched to bypass it.
        markers = [[[f"marker_{i}"]] for i in range(len(flights))]
        return (
            search,
            patch.object(search.client, "post", return_value=response),
            patch(
                "fli.search.flights.json.loads",
                side_effect=[
                    [[None, None, "encoded"]],
                    [None, None, [markers], None],
                ],
            ),
            patch.object(
                SearchFlights, "_parse_flights_data", side_effect=flights
            ),
        )

    def test_round_trip_returns_outbound_list_when_flag_set(
        self, round_trip_search_params
    ):
        """With ``return_combined_only=True``, round-trip search skips recursion."""
        outbound_flights = [
            self._flight_result(500.0),
            self._flight_result(600.0),
            self._flight_result(700.0),
        ]
        search, client_patch, json_patch, parse_patch = self._patched_search(
            outbound_flights
        )
        with client_patch, json_patch, parse_patch:
            results = search.search(
                round_trip_search_params, return_combined_only=True
            )
        assert results == outbound_flights
        # Flat list of outbounds, not (outbound, return) tuples.
        assert all(not isinstance(r, tuple) for r in results)

    def test_one_way_ignores_flag(self, basic_search_params):
        """``return_combined_only`` is a no-op for one-way searches."""
        one_way_flights = [self._flight_result(250.0)]
        search, client_patch, json_patch, parse_patch = self._patched_search(
            one_way_flights
        )
        with client_patch, json_patch, parse_patch:
            results = search.search(basic_search_params, return_combined_only=True)
        assert results == one_way_flights


class TestSelectedReturnSegments:
    """Tests for explicit open-jaw selected-return segment construction."""

    def test_build_selected_return_segments_preserves_open_jaw_route(self):
        today = datetime.now()
        outbound_date = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        return_date = (today + timedelta(days=37)).strftime("%Y-%m-%d")
        selected_outbound = FlightResult(
            legs=[
                FlightLeg(
                    airline=Airline.UA,
                    flight_number="100",
                    departure_airport=Airport.SFO,
                    arrival_airport=Airport.JFK,
                    departure_datetime=datetime.fromisoformat(f"{outbound_date}T08:00:00"),
                    arrival_datetime=datetime.fromisoformat(f"{outbound_date}T16:00:00"),
                    duration=300,
                )
            ],
            price=0.0,
            duration=300,
            stops=0,
        )

        segments, trip_type = build_selected_return_segments(
            outbound_origin=Airport.SFO,
            outbound_destination=Airport.JFK,
            outbound_date=outbound_date,
            selected_outbound=selected_outbound,
            return_origin=Airport.LGA,
            return_destination=Airport.SFO,
            return_date=return_date,
        )
        filters = FlightSearchFilters(
            trip_type=trip_type,
            passenger_info=PassengerInfo(adults=1),
            flight_segments=segments,
        )
        formatted_segments = filters.format()[1][13]

        assert formatted_segments[0][8] == [
            ["SFO", outbound_date, "JFK", None, "UA", "100"]
        ]
        assert formatted_segments[1][0] == [[["LGA", 0]]]
        assert formatted_segments[1][1] == [[["SFO", 0]]]
        assert formatted_segments[1][6] == return_date


class TestFareOptionsInspection:
    """Tests for typed fare-options inspection results."""

    def test_inspect_fare_options_reports_unavailable_when_payload_has_no_fares(
        self, basic_search_params
    ):
        search = SearchFlights()
        with (
            patch.object(search, "_post_and_extract_payload", return_value=json.dumps([None])),
        ):
            result = search.inspect_fare_options(basic_search_params)

        assert isinstance(result, FareOptionsResult)
        assert result.available is False
        assert result.fare_options == []
        assert result.reason == "fare_options_not_exposed_in_google_shopping_payload"

    def test_inspect_fare_options_extracts_explicit_fare_price(self, basic_search_params):
        search = SearchFlights()
        payload = [["Economy Flexible", [[None, 249]], "fare family"]]

        with patch.object(search, "_post_and_extract_payload", return_value=json.dumps(payload)):
            result = search.inspect_fare_options(basic_search_params)

        assert result.available is True
        assert result.fare_options[0].brand == "Economy Flexible"
        assert result.fare_options[0].price == 249.0
        assert result.fare_options[0].cabin == "economy"


class TestParsePriceInfo:
    """Tests for _parse_price_info method handling missing/malformed price data."""

    def test_parse_price_info_valid_data(self):
        """Test _parse_price_info with valid price data."""
        data = [None, [[100, 200, 299.99]]]
        price, currency = SearchFlights._parse_price_info(data)
        assert price == 299.99
        assert currency is None

    def test_parse_price_info_empty_inner_list(self):
        """Test _parse_price_info returns 0.0 when inner price list is empty."""
        data = [None, [[]]]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_empty_outer_list(self):
        """Test _parse_price_info returns 0.0 when outer price list is empty."""
        data = [None, []]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_none_price_section(self):
        """Test _parse_price_info returns 0.0 when price section is None."""
        data = [None, None]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_missing_price_section(self):
        """Test _parse_price_info returns 0.0 when data has no price section."""
        data = [None]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_inner_list_none(self):
        """Test _parse_price_info returns 0.0 when inner list is None."""
        data = [None, [None]]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_currency_from_live_price_token(self):
        """_parse_currency should decode the returned currency from a live token sample."""
        data = [
            None,
            [
                [None, 118],
                "CjRIQktCNmV1UjNqNjhBR043X0FCRy0tLS0tLS0tLS12dGpkN0FBQUFBR25JcWZNS2pGTTBBEgZV"
                "QTIyMDkaCgjcWxACGgNVU0Q4HHDcWw==",
            ],
        ]
        assert SearchFlights._parse_currency(data) == "USD"

    def test_parse_price_info_combines_price_and_currency(self):
        """_parse_price_info should preserve price and extract the returned currency."""
        data = [
            None,
            [
                [None, 118],
                "CjRIQktCNmV1UjNqNjhBR043X0FCRy0tLS0tLS0tLS12dGpkN0FBQUFBR25JcWZNS2pGTTBBEgZV"
                "QTIyMDkaCgjcWxACGgNVU0Q4HHDcWw==",
            ],
        ]
        assert SearchFlights._parse_price_info(data) == (118.0, "USD")


class TestEmptyResponseDiagnostics:
    """Tests for the empty-payload diagnostic helpers."""

    def test_extract_proto_type_marker_finds_error_response(self):
        from fli.search.flights import _extract_proto_type_marker

        body = (
            ")]}'\n\n[[\"wrb.fr\",null,null,null,null,[13,null,["
            "[\"type.googleapis.com/travel.frontend.flights.ErrorResponse\","
            "[[null,[[0,0,0],null,null,null,null,[[0]]],0,\"abc\"],0]]]]]]"
        )
        assert (
            _extract_proto_type_marker(body)
            == "type.googleapis.com/travel.frontend.flights.ErrorResponse"
        )

    def test_extract_proto_type_marker_none_when_absent(self):
        from fli.search.flights import _extract_proto_type_marker

        assert _extract_proto_type_marker(")]}'\n\n[[\"wrb.fr\",null]]") is None
        assert _extract_proto_type_marker("") is None
        assert _extract_proto_type_marker("<html>captcha</html>") is None

    def test_describe_json_shape_list_first_items(self):
        from fli.search.flights import _describe_json_shape

        assert _describe_json_shape([1, "x", None, True, {}]).startswith("list[5](")
        assert _describe_json_shape([]) == "list[0]"

    def test_describe_json_shape_scalars(self):
        from fli.search.flights import _describe_json_shape

        assert _describe_json_shape(None) == "null"
        assert _describe_json_shape(True) == "bool"
        assert _describe_json_shape(3.14) == "number"
        assert _describe_json_shape("hello") == "str(len=5)"

    def test_parse_inner_payload_returns_none_on_shape_mismatch(self):
        assert SearchFlights._parse_inner_payload(")]}'\n[]") is None
        assert SearchFlights._parse_inner_payload(")]}'\n[[]]") is None
        assert SearchFlights._parse_inner_payload("<html>") is None

    def test_parse_inner_payload_returns_none_when_inner_null(self):
        assert SearchFlights._parse_inner_payload(")]}'\n[[\"a\", null, null]]") is None

    def test_parse_inner_payload_returns_string_when_present(self):
        raw = ")]}'\n[[\"a\", null, \"[1,2,3]\"]]"
        assert SearchFlights._parse_inner_payload(raw) == "[1,2,3]"

    def test_is_deterministic_error_response_flags_error_response(self):
        from fli.search.flights import _is_deterministic_error_response

        body = (
            '[["wrb.fr",null,null,null,null,[13,null,'
            '[["type.googleapis.com/travel.frontend.flights.ErrorResponse",[]]]]]]'
        )
        assert _is_deterministic_error_response(body) is True

    def test_is_deterministic_error_response_ignores_unknown_protos(self):
        from fli.search.flights import _is_deterministic_error_response

        body = '"type.googleapis.com/some.other.Type"'
        assert _is_deterministic_error_response(body) is False

    def test_is_deterministic_error_response_false_when_no_proto(self):
        from fli.search.flights import _is_deterministic_error_response

        assert _is_deterministic_error_response(")]}'\n[[\"wrb.fr\"]]") is False
        assert _is_deterministic_error_response("") is False


class TestFailFastOnErrorResponse:
    """Empty responses carrying an ErrorResponse proto should not retry."""

    @staticmethod
    def _error_response_body() -> str:
        return (
            ")]}'\n[[\"wrb.fr\",null,null,null,null,[13,null,"
            "[[\"type.googleapis.com/travel.frontend.flights.ErrorResponse\","
            "[[null,[[0,0,0],null,null,null,null,[[0]]],0,\"x\"],0]]]]]]"
        )

    def test_fail_fast_returns_none_without_retries(self):
        """First attempt returns ErrorResponse → no second attempt fired."""
        search = SearchFlights()
        response = MagicMock()
        response.text = self._error_response_body()
        response.status_code = 200
        with patch.object(search.client, "post", return_value=response) as post_spy:
            result = search._post_and_extract_payload("f=encoded")
        assert result is None
        assert post_spy.call_count == 1

    def test_transient_empty_still_retries(self):
        """Empty body without ErrorResponse proto keeps retrying."""
        search = SearchFlights()
        response = MagicMock()
        # Valid outer JSON with null inner — no proto marker.
        response.text = ")]}'\n[[\"wrb.fr\", null, null]]"
        response.status_code = 200
        with patch.object(
            search.client, "post", return_value=response
        ) as post_spy, patch.object(
            search.client, "warm_up_session", return_value=True
        ):
            result = search._post_and_extract_payload("f=encoded")
        assert result is None
        # All retries attempted — empty without ErrorResponse is assumed transient.
        assert post_spy.call_count == 3


class TestTravelWarningHandling:
    """Travel-restriction advisories injected by Google must not crash parsing."""

    @staticmethod
    def _warning_entry() -> list:
        return [
            12,
            None,
            None,
            None,
            None,
            ["Travel restricted", "Airspace closure may affect flights.", 2],
        ]

    @staticmethod
    def _itinerary_entry(price: float = 500.0) -> list:
        # Minimal flight blob shaped like _parse_flights_data expects:
        #   data[0][9] = duration (minutes)
        #   data[0][2] = legs list (one leg = nonstop)
        #   data[1] = price block; data[1][0][-1] = price
        leg = [None] * 23
        leg[3] = "WAW"
        leg[6] = "HEL"
        leg[8] = [10, 30]
        leg[10] = [13, 0]
        leg[11] = 150
        leg[20] = [2026, 6, 12]
        leg[21] = [2026, 6, 12]
        leg[22] = ["AY", "100"]

        main = [None] * 14
        main[2] = [leg]
        main[9] = 150

        return [
            main,
            [[None, None, price], "USD"],
        ]

    def _stub_search(self, encoded_payload: list) -> SearchFlights:
        from fli.models import (
            FlightSearchFilters,
            FlightSegment,
            MaxStops,
            PassengerInfo,
            SeatType,
            SortBy,
        )
        from fli.models.google_flights.base import TripType

        search = SearchFlights()
        # Wire _post_and_extract_payload to return our crafted JSON.
        with patch.object(
            search,
            "_post_and_extract_payload",
            return_value=__import__("json").dumps(encoded_payload),
        ):
            filters = FlightSearchFilters(
                passenger_info=PassengerInfo(
                    adults=1, children=0, infants_in_seat=0, infants_on_lap=0
                ),
                flight_segments=[
                    FlightSegment(
                        departure_airport=[[Airport.WAW, 0]],
                        arrival_airport=[[Airport.HEL, 0]],
                        travel_date=(
                            datetime.now() + timedelta(days=30)
                        ).strftime("%Y-%m-%d"),
                    )
                ],
                stops=MaxStops.ANY,
                seat_type=SeatType.ECONOMY,
                sort_by=SortBy.CHEAPEST,
                trip_type=TripType.ONE_WAY,
            )
            results = search.search(filters)
        return search, results

    def test_inline_warning_does_not_crash_parser(self):
        """Repro: advisory entry inline in section [2][0]. Previously TypeError'd."""
        payload = [None, None, [[self._warning_entry(), self._itinerary_entry()]], None]
        search, results = self._stub_search(payload)

        assert results is not None
        assert len(results) == 1
        assert results[0].price == 500.0
        assert search.last_warnings, "the inline advisory should surface"
        assert search.last_warnings[0].title == "Travel restricted"
        assert search.last_warnings[0].code == 12
        assert search.last_warnings[0].severity == 2

    def test_top_level_warning_at_data_22_is_collected(self):
        """Advisory placed at top-level data[22] is collected, parsing unaffected."""
        payload = [None] * 31
        payload[2] = [[self._itinerary_entry()]]
        payload[22] = [self._warning_entry()]
        search, results = self._stub_search(payload)

        assert results is not None
        assert len(results) == 1
        titles = [w.title for w in search.last_warnings]
        assert "Travel restricted" in titles

    def test_clean_response_has_empty_warnings(self):
        payload = [None, None, [[self._itinerary_entry()]], None]
        search, results = self._stub_search(payload)

        assert results is not None
        assert search.last_warnings == []

    def test_is_itinerary_entry_discriminator(self):
        from fli.search.flights import _is_itinerary_entry

        assert _is_itinerary_entry(self._itinerary_entry()) is True
        assert _is_itinerary_entry(self._warning_entry()) is False
        assert _is_itinerary_entry([]) is False
        assert _is_itinerary_entry(None) is False
        assert _is_itinerary_entry("a string") is False

    def test_parse_travel_warning_extracts_fields(self):
        from fli.search.flights import _parse_travel_warning

        parsed = _parse_travel_warning(self._warning_entry())
        assert parsed is not None
        assert parsed.code == 12
        assert parsed.title == "Travel restricted"
        assert parsed.message == "Airspace closure may affect flights."
        assert parsed.severity == 2

    def test_parse_travel_warning_returns_none_for_unrelated(self):
        from fli.search.flights import _parse_travel_warning

        assert _parse_travel_warning([1, 2, 3]) is None
        assert _parse_travel_warning("not a list") is None
        assert _parse_travel_warning([]) is None

    def test_is_itinerary_entry_rejects_empty_inner_list(self):
        """Edge case: real-looking entry with empty el[0] should be filtered.

        Without this guard, _parse_flights_data would crash with IndexError
        on data[0][9] — distinct from the original int-marker bug but in
        the same crash class.
        """
        from fli.search.flights import _is_itinerary_entry

        assert _is_itinerary_entry([[], None]) is False

    def test_round_trip_recursion_preserves_outbound_warnings(
        self, round_trip_search_params
    ):
        """Outer-call advisories must survive recursive return-leg searches.

        The round-trip flow recurses to fetch each outbound's return
        options. Each recursive search() overwrites self.last_warnings.
        The outer call must restore the outbound advisories before
        returning so callers see the warnings for the search they issued.
        """
        from fli.models.google_flights.base import TripType

        outbound_warning = self._warning_entry()
        outbound_itinerary = self._itinerary_entry(price=400.0)
        return_itinerary = self._itinerary_entry(price=300.0)
        outbound_payload = [
            None,
            None,
            [[outbound_warning, outbound_itinerary]],
            None,
        ]
        # Return-leg response carries no warnings — that's the case
        # where the outer-call advisory must survive.
        return_payload = [None, None, [[return_itinerary]], None]

        search = SearchFlights()

        import json as _json

        payloads = iter(
            [_json.dumps(outbound_payload), _json.dumps(return_payload)]
        )

        with patch.object(
            search,
            "_post_and_extract_payload",
            side_effect=lambda *_a, **_kw: next(payloads),
        ):
            params = round_trip_search_params
            params.trip_type = TripType.ROUND_TRIP
            results = search.search(params, top_n=1)

        assert results is not None
        assert search.last_warnings, "outbound advisory must survive recursion"
        assert search.last_warnings[0].title == "Travel restricted"
