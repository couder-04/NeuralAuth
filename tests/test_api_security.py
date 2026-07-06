"""
tests/test_api_security.py

Tests for the security hardening in api/server.py:

    1. Optional, opt-in API-key auth on POST /authenticate (disabled by
       default so existing/local-dev behavior is unchanged; enabled by
       setting the TRANSACTION_ENGINE_API_KEY env var).
    2. Unhandled exceptions never leak internal details (str(e), stack
       traces, file paths, ...) to the HTTP client -- only a generic
       message + an opaque request_id, with the real exception logged
       server-side instead.
"""

import pytest
from fastapi.testclient import TestClient

from api import server


client = TestClient(server.app)


def _minimal_payload():
    return {
        "user_id": "USR_1001",
        "transcript": "Check my balance",
    }


# ==========================================================
# API key is off by default (backward compatible)
# ==========================================================

def test_authenticate_is_unauthenticated_by_default(monkeypatch):
    monkeypatch.delenv(server.API_KEY_ENV_VAR, raising=False)

    response = client.post("/authenticate", json=_minimal_payload())

    # No API key header supplied, none required -- must not be 401.
    assert response.status_code != 401


# ==========================================================
# API key enforcement when configured
# ==========================================================

def test_authenticate_rejects_missing_key_when_configured(monkeypatch):
    monkeypatch.setenv(server.API_KEY_ENV_VAR, "correct-key")

    response = client.post("/authenticate", json=_minimal_payload())

    assert response.status_code == 401


def test_authenticate_rejects_wrong_key_when_configured(monkeypatch):
    monkeypatch.setenv(server.API_KEY_ENV_VAR, "correct-key")

    response = client.post(
        "/authenticate",
        json=_minimal_payload(),
        headers={"x-api-key": "wrong-key"},
    )

    assert response.status_code == 401


def test_authenticate_accepts_correct_key_when_configured(monkeypatch):
    monkeypatch.setenv(server.API_KEY_ENV_VAR, "correct-key")

    response = client.post(
        "/authenticate",
        json=_minimal_payload(),
        headers={"x-api-key": "correct-key"},
    )

    # Correct key clears the auth gate; whatever happens next depends
    # on the (unmocked) real pipeline, but it must not be an auth
    # rejection.
    assert response.status_code != 401


def test_health_and_root_never_require_api_key(monkeypatch):
    """Only /authenticate is gated -- health checks must stay reachable
    for load balancers / orchestrators regardless of auth config."""
    monkeypatch.setenv(server.API_KEY_ENV_VAR, "correct-key")

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_require_api_key_dependency_is_a_noop_without_configured_key():
    """Direct unit test of the dependency function itself."""
    import os

    os.environ.pop(server.API_KEY_ENV_VAR, None)
    # Should not raise regardless of what header value is passed.
    server.require_api_key(x_api_key=None)
    server.require_api_key(x_api_key="anything")


# ==========================================================
# Unhandled exceptions never leak internals to the client
# ==========================================================

def test_unhandled_exception_does_not_leak_internal_details(monkeypatch):
    monkeypatch.delenv(server.API_KEY_ENV_VAR, raising=False)

    secret_internal_detail = "Feature order mismatch between model_info and feature_info."

    def broken_predictor():
        class BrokenPredictor:
            def predict_result(self, request):
                raise RuntimeError(secret_internal_detail)

        return BrokenPredictor()

    monkeypatch.setattr(server, "get_predictor", broken_predictor)

    response = client.post("/authenticate", json=_minimal_payload())

    assert response.status_code == 500
    body = response.json()
    assert secret_internal_detail not in body["detail"]
    assert "request_id=" in body["detail"]


def test_unhandled_exception_is_logged_server_side(monkeypatch, caplog):
    monkeypatch.delenv(server.API_KEY_ENV_VAR, raising=False)

    def broken_predictor():
        class BrokenPredictor:
            def predict_result(self, request):
                raise RuntimeError("boom-internal-detail")

        return BrokenPredictor()

    monkeypatch.setattr(server, "get_predictor", broken_predictor)

    with caplog.at_level("ERROR"):
        client.post("/authenticate", json=_minimal_payload())

    assert "boom-internal-detail" in caplog.text
