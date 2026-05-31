"""Models for interacting with Google Flights API.

This module contains all the data models used for flight searches and results.
Models are designed to match Google Flights' APIs while providing a clean pythonic interface.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from fli.models.airline import Airline
from fli.models.airport import Airport

KGMID_PATTERN = re.compile(r"^/m/[A-Za-z0-9_]+$")


class SeatType(Enum):
    """Available cabin classes for flights."""

    ECONOMY = 1
    PREMIUM_ECONOMY = 2
    BUSINESS = 3
    FIRST = 4


class SortBy(Enum):
    """Available sorting options for flight results.

    Maps to the top-level sort_mode value in the Google Flights API payload.
    """

    TOP_FLIGHTS = 0
    BEST = 1
    CHEAPEST = 2
    DEPARTURE_TIME = 3
    ARRIVAL_TIME = 4
    DURATION = 5
    EMISSIONS = 6


class TripType(Enum):
    """Type of flight journey."""

    ROUND_TRIP = 1
    ONE_WAY = 2
    MULTI_CITY = 3


class MaxStops(Enum):
    """Maximum number of stops allowed in flight search."""

    ANY = 0
    NON_STOP = 1
    ONE_STOP_OR_FEWER = 2
    TWO_OR_FEWER_STOPS = 3


class EmissionsFilter(Enum):
    """Filter flights by carbon emissions level.

    Corresponds to the "Less emissions" toggle on Google Flights.
    When enabled, only flights with lower-than-average CO2 emissions are shown.
    """

    ALL = 0
    LESS = 1


class Currency(Enum):
    """Supported currencies for pricing. Currently only USD."""

    USD = "USD"
    # Placeholder for other currencies


class BagsFilter(BaseModel):
    """Include checked/carry-on bag fees in displayed prices.

    When set, Google Flights adjusts the displayed price to include baggage costs,
    making comparisons between budget and full-service carriers fairer.
    """

    checked_bags: NonNegativeInt = 0
    carry_on: bool = False


class TimeRestrictions(BaseModel):
    """Time constraints for flight departure and arrival in local time.

    All times are in hours from midnight (e.g., 20 = 8:00 PM).
    """

    earliest_departure: NonNegativeInt | None = None
    latest_departure: PositiveInt | None = None
    earliest_arrival: NonNegativeInt | None = None
    latest_arrival: PositiveInt | None = None

    @field_validator("latest_departure", "latest_arrival")
    @classmethod
    def validate_latest_times(
        cls, v: PositiveInt | None, info: ValidationInfo
    ) -> PositiveInt | None:
        """Validate and adjust the latest time restrictions."""
        if v is None:
            return v

        # Get "departure" or "arrival" from field name
        field_prefix = "earliest_" + info.field_name[7:]
        earliest = info.data.get(field_prefix)

        # Swap values to ensure that `from` is always before `to`
        if earliest is not None and earliest > v:
            info.data[field_prefix] = v
            return earliest
        return v


class PassengerInfo(BaseModel):
    """Passenger configuration for flight search."""

    adults: NonNegativeInt = 1
    children: NonNegativeInt = 0
    infants_in_seat: NonNegativeInt = 0
    infants_on_lap: NonNegativeInt = 0


class PriceLimit(BaseModel):
    """Maximum price constraint for flight search."""

    max_price: PositiveInt
    currency: Currency | None = Currency.USD


class LayoverRestrictions(BaseModel):
    """Constraints for layovers in multi-leg flights."""

    airports: list[Airport] | None = None
    max_duration: PositiveInt | None = None


class FlightLeg(BaseModel):
    """A single flight leg (segment) with airline and timing details."""

    airline: Airline
    flight_number: str
    departure_airport: Airport
    arrival_airport: Airport
    departure_datetime: datetime
    arrival_datetime: datetime
    duration: PositiveInt  # in minutes


class FlightResult(BaseModel):
    """Complete flight search result with pricing and timing."""

    legs: list[FlightLeg]
    price: NonNegativeFloat  # in specified currency
    currency: str | None = None
    duration: PositiveInt  # total duration in minutes
    stops: NonNegativeInt
    raw_data: Any | None = Field(default=None, exclude=True, repr=False)


class FareOption(BaseModel):
    """A fare-family option for a selected itinerary, when Google exposes one."""

    brand: str | None = None
    cabin: str | None = None
    basic_economy: bool | None = None
    price: NonNegativeFloat | None = None
    currency: str | None = None
    bags: str | None = None
    refundability: str | None = None
    changeability: str | None = None
    raw_path: str | None = None


class FareOptionsResult(BaseModel):
    """Result of inspecting Google payloads for fare-family pricing."""

    available: bool
    fare_options: list[FareOption] = Field(default_factory=list)
    reason: str | None = None


class TravelWarning(BaseModel):
    """Travel advisory dialog Google injects into the response.

    Observed on routes affected by airspace closures (e.g. Warsaw to Asia
    after Russian airspace was closed): Google emits a sibling entry such
    as ``[12, null, null, null, null, ["Travel restricted",
    "Airspace closure may affect flights.", 2]]`` either at the top level
    of the parsed payload or inline as a sibling of itineraries inside
    ``data[2][0]``/``data[3][0]``.

    The leading int is a type marker (``12`` = travel-restriction
    advisory) and the trailing int in the body is a severity (1=info,
    2=warning).

    Surfaced via :attr:`SearchFlights.last_warnings` so callers can read
    advisories without parsing the raw payload themselves.
    """

    code: int
    title: str
    message: str
    severity: int


class FlightSegment(BaseModel):
    """A segment represents a single portion of a flight journey between two airports.

    For example, in a one-way flight from JFK to LAX, there would be one segment.
    In a multi-city trip from JFK -> LAX -> SEA, there would be two segments:
    JFK -> LAX and LAX -> SEA.

    Airport endpoints accept either :class:`Airport` enum values for IATA codes
    or Google Knowledge Graph identifier strings (kgmid, e.g. ``"/m/02_286"``)
    which represent city-level searches on Google Flights.

    The second element of each entry is a slot index. For IATA airports, use
    ``0``. For kgmid city ids, use ``5`` — Google's payload rejects kgmids with
    slot ``0`` and returns empty results. The :func:`fli.core.location_entry`
    helper picks the correct slot automatically.
    """

    departure_airport: list[list[Airport | str | int]]
    arrival_airport: list[list[Airport | str | int]]
    travel_date: str
    time_restrictions: TimeRestrictions | None = None
    selected_flight: FlightResult | None = None

    @property
    def parsed_travel_date(self) -> datetime:
        """Parse the travel date string into a datetime object."""
        return datetime.strptime(self.travel_date, "%Y-%m-%d")

    @field_validator("travel_date")
    @classmethod
    def validate_travel_date(cls, v: str) -> str:
        """Validate that the travel date is not in the past."""
        travel_date = datetime.strptime(v, "%Y-%m-%d").date()
        if travel_date < datetime.now().date():
            raise ValueError("Travel date cannot be in the past")
        return v

    @field_validator("departure_airport", "arrival_airport")
    @classmethod
    def validate_airport_entries(
        cls, entries: list[list[Airport | str | int]]
    ) -> list[list[Airport | str | int]]:
        """Validate each airport entry is ``[Airport|kgmid_str, slot_int]``.

        Strings must match the kgmid pattern (e.g. ``"/m/02_286"``). Arbitrary
        strings — including IATA codes passed as strings — are rejected; use
        the :class:`Airport` enum for IATA codes instead.
        """
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(
                    f"Airport entry must be [airport, slot_index], got {entry!r}"
                )
            airport, slot = entry[0], entry[1]
            if isinstance(airport, Airport):
                pass
            elif isinstance(airport, str):
                if not KGMID_PATTERN.match(airport):
                    raise ValueError(
                        "Airport string must be a kgmid like '/m/02_286'; "
                        f"use Airport enum for IATA codes. Got: {airport!r}"
                    )
            else:
                raise ValueError(
                    "Airport must be an Airport enum member or kgmid string, "
                    f"got: {type(airport).__name__}"
                )
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise ValueError(f"Slot index must be int, got: {type(slot).__name__}")
        return entries

    @model_validator(mode="after")
    def validate_airports(self) -> "FlightSegment":
        """Validate that departure and arrival airports are different."""
        if not self.departure_airport or not self.arrival_airport:
            raise ValueError("Both departure and arrival airports must be specified")

        dep_airport = self.departure_airport[0][0]
        arr_airport = self.arrival_airport[0][0]

        # Compare only when both endpoints are the same kind (Airport vs kgmid);
        # otherwise they are inherently different.
        if isinstance(dep_airport, Airport) and isinstance(arr_airport, Airport):
            if dep_airport == arr_airport:
                raise ValueError("Departure and arrival airports must be different")
        elif isinstance(dep_airport, str) and isinstance(arr_airport, str):
            if dep_airport == arr_airport:
                raise ValueError("Departure and arrival airports must be different")
        return self
