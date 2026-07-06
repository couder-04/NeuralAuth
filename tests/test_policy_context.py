"""
tests/test_policy_context.py

Unit tests for engines/policy_context.py -- the module that derives the
categorical `PolicyInput.location_familiarity` / `time_familiarity`
labels from the continuous 0..1 FeatureVector scores (and, for time,
optionally the request's own wall-clock timestamp).
"""

from datetime import datetime

from engines.policy_context import (
    classify_location_familiarity,
    classify_time_familiarity,
    parse_request_timestamp,
)


# ---------------------------------------------------------
# Location familiarity
# ---------------------------------------------------------

def test_location_at_or_above_threshold_is_familiar():
    assert classify_location_familiarity(0.5) == "FAMILIAR"
    assert classify_location_familiarity(0.95) == "FAMILIAR"
    assert classify_location_familiarity(1.0) == "FAMILIAR"


def test_location_below_threshold_is_unfamiliar():
    assert classify_location_familiarity(0.49) == "UNFAMILIAR"
    assert classify_location_familiarity(0.0) == "UNFAMILIAR"


def test_location_threshold_is_configurable():
    assert classify_location_familiarity(0.3, threshold=0.2) == "FAMILIAR"
    assert classify_location_familiarity(0.3, threshold=0.4) == "UNFAMILIAR"


# ---------------------------------------------------------
# Time familiarity -- score-only (no timestamp)
# ---------------------------------------------------------

def test_time_at_or_above_threshold_with_no_timestamp_is_normal():
    assert classify_time_familiarity(0.5) == "NORMAL"
    assert classify_time_familiarity(0.9) == "NORMAL"


def test_time_below_threshold_is_odd_hour_regardless_of_timestamp():
    assert classify_time_familiarity(0.1) == "ODD_HOUR"


# ---------------------------------------------------------
# Time familiarity -- timestamp signal
# ---------------------------------------------------------

def test_high_familiarity_but_odd_clock_hour_is_still_odd_hour():
    # 03:00 falls inside the default (0, 5) "objectively unusual" window,
    # even though the user's own familiarity score is high.
    odd_hour_ts = datetime(2026, 1, 1, 3, 0, 0)
    assert classify_time_familiarity(0.95, timestamp=odd_hour_ts) == "ODD_HOUR"


def test_high_familiarity_and_normal_clock_hour_is_normal():
    normal_ts = datetime(2026, 1, 1, 14, 30, 0)
    assert classify_time_familiarity(0.95, timestamp=normal_ts) == "NORMAL"


def test_missing_timestamp_falls_back_to_score_only():
    assert classify_time_familiarity(0.95, timestamp=None) == "NORMAL"


def test_odd_hour_range_is_configurable():
    ts = datetime(2026, 1, 1, 22, 0, 0)
    assert classify_time_familiarity(0.95, timestamp=ts) == "NORMAL"
    assert (
        classify_time_familiarity(0.95, timestamp=ts, odd_hour_range=(21, 23))
        == "ODD_HOUR"
    )


# ---------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------

def test_parse_request_timestamp_accepts_iso8601():
    parsed = parse_request_timestamp("2026-01-01T03:00:00")
    assert parsed == datetime(2026, 1, 1, 3, 0, 0)


def test_parse_request_timestamp_none_when_missing():
    assert parse_request_timestamp(None) is None
    assert parse_request_timestamp("") is None


def test_parse_request_timestamp_none_when_unparseable():
    """Never raises -- a bad timestamp just means "no wall-clock signal",
    not a hard failure; the familiarity score is still usable on its
    own."""
    assert parse_request_timestamp("not-a-timestamp") is None
