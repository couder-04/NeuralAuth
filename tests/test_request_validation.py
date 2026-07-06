"""
tests/test_request_validation.py

Tests for the input-validation hardening added to
models/request.py::TransactionRequest.

Before this, `TransactionRequest` accepted essentially anything
(blank user_id/transcript, GPS coordinates outside the -90..90 /
-180..180 range, negative vehicle speed, unparseable timestamps) and
let malformed values crash deep inside FeatureExtractor/PolicyEngine
as raw, uncaught exceptions -- rather than being rejected cleanly at
the API boundary with a 422.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.server import app
from models.request import TransactionRequest


client = TestClient(app)


def _minimal(**overrides):
    payload = {"user_id": "USR_1001", "transcript": "Check my balance"}
    payload.update(overrides)
    return payload


# ==========================================================
# Valid requests still work (backward compatibility)
# ==========================================================

def test_minimal_valid_request_is_accepted():
    TransactionRequest(**_minimal())


def test_full_valid_request_is_accepted():
    TransactionRequest(
        **_minimal(
            gps_latitude=12.9716,
            gps_longitude=77.5946,
            vehicle_speed=42.0,
            timestamp="2026-01-01T10:30:00",
        )
    )


# ==========================================================
# user_id / transcript must not be blank
# ==========================================================

def test_empty_user_id_is_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(user_id=""))


def test_whitespace_only_user_id_is_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(user_id="   "))


def test_empty_transcript_is_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(transcript=""))


# ==========================================================
# GPS coordinates must be within valid ranges
# ==========================================================

@pytest.mark.parametrize("latitude", [-90.0, 0.0, 90.0])
def test_valid_latitude_boundaries_accepted(latitude):
    TransactionRequest(**_minimal(gps_latitude=latitude))


@pytest.mark.parametrize("latitude", [90.0001, -90.0001, 999.0, -999.0])
def test_invalid_latitude_rejected(latitude):
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(gps_latitude=latitude))


@pytest.mark.parametrize("longitude", [-180.0, 0.0, 180.0])
def test_valid_longitude_boundaries_accepted(longitude):
    TransactionRequest(**_minimal(gps_longitude=longitude))


@pytest.mark.parametrize("longitude", [180.0001, -180.0001, 999.0])
def test_invalid_longitude_rejected(longitude):
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(gps_longitude=longitude))


# ==========================================================
# vehicle_speed must be non-negative
# ==========================================================

def test_negative_vehicle_speed_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(vehicle_speed=-1.0))


def test_zero_vehicle_speed_accepted():
    TransactionRequest(**_minimal(vehicle_speed=0.0))


# ==========================================================
# timestamp must be a parseable ISO-8601 string, if provided
# ==========================================================

def test_missing_timestamp_is_fine():
    TransactionRequest(**_minimal())


def test_valid_iso8601_timestamp_accepted():
    TransactionRequest(**_minimal(timestamp="2026-01-01T03:00:00"))


def test_malformed_timestamp_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**_minimal(timestamp="not-a-timestamp"))


# ==========================================================
# End-to-end: the API returns a clean 422, not a 500 crash
# ==========================================================

def test_api_rejects_invalid_gps_with_422_not_500():
    response = client.post(
        "/authenticate",
        json=_minimal(gps_latitude=999.0),
    )
    assert response.status_code == 422


def test_api_rejects_blank_user_id_with_422_not_500():
    response = client.post(
        "/authenticate",
        json=_minimal(user_id=""),
    )
    assert response.status_code == 422


def test_api_rejects_malformed_timestamp_with_422_not_500():
    response = client.post(
        "/authenticate",
        json=_minimal(timestamp="definitely-not-a-date"),
    )
    assert response.status_code == 422
