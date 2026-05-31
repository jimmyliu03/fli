"""Tests for Google Flights booking-page fare extraction."""

import json
from html import escape

from fli.search.booking import (
    build_booking_results_request_body,
    extract_booking_constraints,
    parse_booking_results_fare_options,
)


def test_extract_booking_constraints_from_p4ikpb_data_p():
    constraints = [
        None,
        None,
        3,
        None,
        [],
        1,
        [3, 0, 0, 0],
        None,
        None,
        None,
        None,
        None,
        None,
        [
            [
                [[["IND", 0]]],
                [[["/m/0ftjx", 5]]],
                None,
                0,
                None,
                None,
                "2026-07-04",
                None,
                [["IND", "2026-07-04", "ORD", None, "UA", "5419"]],
            ]
        ],
    ]
    app_state = [
        [
            [None, None, 0, "token"],
            [3, constraints, None, [None, [0, True], None, []]],
        ]
    ]
    data_p = "%.@." + json.dumps(app_state, separators=(",", ":"))
    html = f'<c-wiz jsrenderer="p4IKPb" data-p="{escape(data_p)}"></c-wiz>'

    assert extract_booking_constraints(html) == constraints


def test_build_booking_results_request_body_uses_string_wrapped_request():
    constraints = [None, None, 3]

    body = build_booking_results_request_body(constraints)

    assert json.loads(body) == [None, json.dumps([None, constraints], separators=(",", ":"))]


def test_parse_booking_results_fare_options():
    options = [
        [
            0,
            [["UA", "United", None, True]],
            None,
            [["UA", "5419"]],
            True,
            None,
            None,
            [[None, 4057], "token"],
            None,
            None,
            False,
            None,
            None,
            None,
            [[[None, ["UA", "BASIC ECONOMY"], 1]]],
            None,
            None,
            [[1, 2, 3], None, None, None, None, [[6]]],
            [[2, [[None, 180]], 1], [3]],
            None,
            None,
            [["UA", "BASIC ECONOMY"], [], True, "Basic Economy"],
        ],
        [
            0,
            [["UA", "United", None, True]],
            None,
            [["UA", "5419"]],
            True,
            None,
            None,
            [[None, 4717], "token"],
            None,
            None,
            False,
            None,
            None,
            None,
            [[[None, ["UA", "ECONOMY"], 1]]],
            None,
            None,
            [[1, 2, 3], None, None, None, None, [[7]]],
            [[3], [3]],
            None,
            None,
            [["UA", "ECONOMY"], [], None, "Economy"],
        ],
        [
            0,
            [["UA", "United", None, True]],
            None,
            [["UA", "5419"]],
            True,
            None,
            None,
            [[None, 5392], "token"],
            None,
            None,
            False,
            None,
            None,
            None,
            [[[None, ["UA", "ECONOMY FULLY REFUNDABLE"], 1]]],
            None,
            None,
            [[1, 2, 3], None, None, None, None, [[8]]],
            [[3], [3]],
            None,
            None,
            [["UA", "ECONOMY FULLY REFUNDABLE"], [], None, "Economy Fully Refundable"],
        ],
    ]
    inner = [None, [options]]
    outer = [["wrb.fr", None, json.dumps(inner, separators=(",", ":"))]]
    chunk = json.dumps(outer, separators=(",", ":"))
    raw_response = f")]}}'\n\n{len(chunk) + 2}\n{chunk}\n"

    parsed = parse_booking_results_fare_options(raw_response, currency="USD")

    assert [(option.brand, option.price, option.currency) for option in parsed] == [
        ("Basic Economy", 4057.0, "USD"),
        ("Economy", 4717.0, "USD"),
        ("Economy Fully Refundable", 5392.0, "USD"),
    ]
    assert parsed[0].basic_economy is True
    assert parsed[1].basic_economy is False
    assert parsed[2].refundability == "refundable"
