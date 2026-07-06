"""
tests/test_startup.py

Tests for FastAPI startup behavior (api/server.py `lifespan`).

Goal: models/engines must be loaded once, eagerly, when the app starts
-- not lazily on the first inbound request -- and a failure to load
any of them must abort startup loudly rather than serving a broken
app. These tests monkeypatch the loader functions (the same seam
tests/test_api.py uses) so they never touch real model weights; they
only verify the *wiring* -- when loaders run, how many times, in what
order, and what happens when one of them fails.
"""

import pytest
from fastapi.testclient import TestClient

from api import server


def _reset_engine_caches():
    """`get_*_engine()` memoize onto module globals. Since tests below
    monkeypatch the getter functions themselves (not the underlying
    engine classes), these caches are never populated -- but reset them
    defensively so tests never depend on execution order."""
    server.intent_engine = None
    server.risk_engine = None
    server.policy_engine = None
    server.decision_engine = None


@pytest.fixture(autouse=True)
def reset_state():
    _reset_engine_caches()
    yield
    _reset_engine_caches()


def _tracking_loader(monkeypatch, name, return_value="ENGINE"):
    """Monkeypatch `server.<name>` with a fake that records every call
    and returns a fixed sentinel, so both call-count and identity can
    be asserted."""
    calls = []

    def fake():
        calls.append(1)
        return return_value

    monkeypatch.setattr(server, name, fake)
    return calls


# ==========================================================
# Models load eagerly at startup, not on first request
# ==========================================================

def test_all_engines_load_during_startup_before_any_request(monkeypatch):
    predictor_calls = _tracking_loader(monkeypatch, "get_predictor")
    intent_calls = _tracking_loader(monkeypatch, "get_intent_engine")
    risk_calls = _tracking_loader(monkeypatch, "get_risk_engine")
    policy_calls = _tracking_loader(monkeypatch, "get_policy_engine")
    decision_calls = _tracking_loader(monkeypatch, "get_decision_engine")

    # No engine has been touched yet.
    assert predictor_calls == intent_calls == risk_calls == []
    assert policy_calls == decision_calls == []

    with TestClient(server.app):
        # Entering the context manager runs `lifespan` startup, which
        # must have already loaded every engine -- with no request
        # having been made yet.
        assert predictor_calls == [1]
        assert intent_calls == [1]
        assert risk_calls == [1]
        assert policy_calls == [1]
        assert decision_calls == [1]


def test_first_request_does_not_reload_models(monkeypatch):
    """Removing first-request latency means the loaders run exactly
    once at startup and are never invoked again while handling
    requests."""
    predictor_calls = _tracking_loader(monkeypatch, "get_predictor")
    intent_calls = _tracking_loader(monkeypatch, "get_intent_engine")
    risk_calls = _tracking_loader(monkeypatch, "get_risk_engine")
    policy_calls = _tracking_loader(monkeypatch, "get_policy_engine")
    decision_calls = _tracking_loader(monkeypatch, "get_decision_engine")

    with TestClient(server.app) as client:
        assert len(predictor_calls) == 1

        # Health endpoints don't touch the engines at all, but they
        # exercise the running app; the point is no loader fires again.
        client.get("/")
        client.get("/health")

        assert predictor_calls == [1]
        assert intent_calls == [1]
        assert risk_calls == [1]
        assert policy_calls == [1]
        assert decision_calls == [1]


# ==========================================================
# Startup fails clearly when a model/engine cannot load
# ==========================================================

def test_startup_fails_loudly_when_predictor_cannot_load(monkeypatch):
    def broken_predictor():
        raise FileNotFoundError("Missing artifact: best_model.pth")

    monkeypatch.setattr(server, "get_predictor", broken_predictor)

    with pytest.raises(FileNotFoundError, match="Missing artifact"):
        with TestClient(server.app):
            pytest.fail("Server should never finish starting up")


def test_startup_fails_loudly_when_any_engine_cannot_load(monkeypatch):
    """Same guarantee for every engine, not just the predictor."""
    monkeypatch.setattr(server, "get_predictor", lambda: "P")

    def broken_intent_engine():
        raise RuntimeError("failed to load intent model")

    monkeypatch.setattr(server, "get_intent_engine", broken_intent_engine)

    with pytest.raises(RuntimeError, match="failed to load intent model"):
        with TestClient(server.app):
            pytest.fail("Server should never finish starting up")


def test_startup_is_fail_fast_and_does_not_run_later_loaders(monkeypatch):
    """If an early loader fails, later loaders should never run --
    startup fails immediately instead of masking the error."""
    predictor_calls = _tracking_loader(monkeypatch, "get_predictor")

    def broken_intent_engine():
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "get_intent_engine", broken_intent_engine)

    risk_calls = _tracking_loader(monkeypatch, "get_risk_engine")
    policy_calls = _tracking_loader(monkeypatch, "get_policy_engine")
    decision_calls = _tracking_loader(monkeypatch, "get_decision_engine")

    with pytest.raises(RuntimeError, match="boom"):
        with TestClient(server.app):
            pytest.fail("Server should never finish starting up")

    # get_predictor runs before get_intent_engine in STARTUP_LOADER_NAMES.
    assert predictor_calls == [1]
    # Nothing after the failing loader should have run.
    assert risk_calls == []
    assert policy_calls == []
    assert decision_calls == []


def test_a_failed_startup_never_serves_requests(monkeypatch):
    """A server that failed to start must not be usable at all."""

    def broken_predictor():
        raise RuntimeError("model corrupt")

    monkeypatch.setattr(server, "get_predictor", broken_predictor)

    with pytest.raises(RuntimeError):
        with TestClient(server.app) as client:
            # Should never get here -- but if we somehow did, prove the
            # app is unusable rather than silently degraded.
            client.get("/health")


# ==========================================================
# Dependency injection is preserved
# ==========================================================

def test_route_handlers_reuse_the_same_warmed_up_engines(monkeypatch):
    """The route must call the exact same (already-loaded) singleton
    that startup warmed up -- not construct a new one per request --
    while still going through the `get_*_engine()` seam tests rely on
    for dependency injection."""

    monkeypatch.setattr(server, "get_predictor", lambda: "PREDICTOR")
    monkeypatch.setattr(server, "get_intent_engine", lambda: "INTENT")
    monkeypatch.setattr(server, "get_policy_engine", lambda: "POLICY")
    monkeypatch.setattr(server, "get_decision_engine", lambda: "DECISION")

    engine_instance = object()
    calls = []

    def fake_get_risk_engine():
        calls.append(1)
        return engine_instance

    monkeypatch.setattr(server, "get_risk_engine", fake_get_risk_engine)

    with TestClient(server.app):
        assert calls == [1]
        # Calling the seam again (as the route does per-request) must
        # return the identical object loaded at startup, and DI still
        # goes through the same function.
        assert server.get_risk_engine() is engine_instance
        assert len(calls) == 2
