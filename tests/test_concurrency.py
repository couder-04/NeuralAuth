"""
tests/test_concurrency.py

Thread-safety tests for the lazy singleton getters:

    api/server.py        -> get_intent_engine, get_risk_engine,
                             get_policy_engine, get_decision_engine
    inference/predictor.py -> get_predictor

Each getter used to do:

    if engine is None:
        engine = Engine()

which is a classic double-checked-locking race: if two threads both
observe `engine is None` before either finishes constructing it, both
construct an instance, the slower one wins the assignment, and the
first one is silently leaked (for the predictor, that's a whole extra
copy of the model briefly loaded into memory).

These tests don't touch real model weights -- they replace each
engine's *class* (the thing the getter calls to construct an instance)
with a slow dummy that sleeps for a moment before returning, which
reliably widens the race window, then hammer the getter from many
threads at once and assert:

    1. The class was only ever instantiated once.
    2. Every thread observed the exact same singleton instance.
    3. The getter still returns promptly (no deadlock / hang).
"""

import threading
import time

import pytest

from api import server
from inference import predictor as predictor_module


N_THREADS = 32


def _run_concurrently(fn, n=N_THREADS, timeout=10.0):
    """Call `fn()` from `n` threads at (as close to) the same time as
    possible, and return the per-thread results in submission order.
    Fails the test outright (rather than hanging the suite) if any
    thread doesn't finish within `timeout` -- our deadlock guard."""
    results = [None] * n
    errors = [None] * n
    start_barrier = threading.Barrier(n)

    def worker(i):
        start_barrier.wait()
        try:
            results[i] = fn()
        except BaseException as exc:  # noqa: BLE001 - surface to main thread
            errors[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    elapsed = time.perf_counter() - t0

    still_alive = [t for t in threads if t.is_alive()]
    assert not still_alive, (
        f"{len(still_alive)}/{n} threads never finished within "
        f"{timeout}s -- possible deadlock in the singleton getter."
    )
    assert elapsed < timeout, "Getter calls took suspiciously long; possible contention bug."

    for exc in errors:
        if exc is not None:
            raise exc

    return results


class _SlowDummyEngine:
    """Stands in for a real engine/model class. Sleeping *during*
    `__init__` (rather than before it) maximizes the window in which a
    second racing thread could see the module global as still `None`
    and start constructing its own instance, which is exactly the bug
    the lock must prevent."""

    construction_count = 0
    _count_lock = threading.Lock()

    def __init__(self):
        time.sleep(0.05)
        with _SlowDummyEngine._count_lock:
            _SlowDummyEngine.construction_count += 1

    @classmethod
    def reset(cls):
        cls.construction_count = 0


# ==========================================================
# api/server.py engine getters
# ==========================================================

@pytest.fixture(autouse=True)
def reset_server_singletons():
    server.intent_engine = None
    server.risk_engine = None
    server.policy_engine = None
    server.decision_engine = None
    _SlowDummyEngine.reset()
    yield
    server.intent_engine = None
    server.risk_engine = None
    server.policy_engine = None
    server.decision_engine = None


@pytest.mark.parametrize(
    "class_attr, cache_attr, getter_name",
    [
        ("IntentEngine", "intent_engine", "get_intent_engine"),
        ("RiskEngine", "risk_engine", "get_risk_engine"),
        ("PolicyEngine", "policy_engine", "get_policy_engine"),
        ("DecisionEngine", "decision_engine", "get_decision_engine"),
    ],
)
def test_only_one_engine_is_constructed_under_concurrent_access(
    monkeypatch, class_attr, cache_attr, getter_name
):
    monkeypatch.setattr(server, class_attr, _SlowDummyEngine)
    getter = getattr(server, getter_name)

    results = _run_concurrently(getter)

    assert _SlowDummyEngine.construction_count == 1, (
        f"{getter_name}() constructed {_SlowDummyEngine.construction_count} "
        "instances under concurrent access; expected exactly 1."
    )
    # Every thread must have received the identical singleton.
    first = results[0]
    assert all(r is first for r in results)
    assert getattr(server, cache_attr) is first


def test_engine_getter_still_lazy_after_lock_added(monkeypatch):
    """Locking must not turn lazy loading into eager loading: nothing
    should be constructed until the getter is actually called."""
    monkeypatch.setattr(server, "RiskEngine", _SlowDummyEngine)

    assert server.risk_engine is None
    assert _SlowDummyEngine.construction_count == 0

    server.get_risk_engine()

    assert _SlowDummyEngine.construction_count == 1


def test_engine_getter_does_not_reconstruct_once_warm(monkeypatch):
    """After the race is resolved, later (sequential or concurrent)
    calls must hit the fast, lock-free path and reuse the same
    instance -- not re-enter construction."""
    monkeypatch.setattr(server, "RiskEngine", _SlowDummyEngine)

    first = server.get_risk_engine()
    assert _SlowDummyEngine.construction_count == 1

    results = _run_concurrently(server.get_risk_engine)

    assert _SlowDummyEngine.construction_count == 1
    assert all(r is first for r in results)


# ==========================================================
# inference/predictor.py -- get_predictor()
# ==========================================================

@pytest.fixture(autouse=True)
def reset_predictor_singleton():
    predictor_module._predictor = None
    _SlowDummyEngine.reset()
    yield
    predictor_module._predictor = None


def test_only_one_predictor_is_constructed_under_concurrent_access(monkeypatch):
    monkeypatch.setattr(
        predictor_module, "AuthenticationPredictor", _SlowDummyEngine
    )

    results = _run_concurrently(predictor_module.get_predictor)

    assert _SlowDummyEngine.construction_count == 1, (
        "get_predictor() constructed "
        f"{_SlowDummyEngine.construction_count} predictors under concurrent "
        "access; expected exactly 1 (this would otherwise double-load the "
        "model into memory)."
    )
    first = results[0]
    assert all(r is first for r in results)
    assert predictor_module._predictor is first


def test_predictor_getter_still_lazy_after_lock_added(monkeypatch):
    monkeypatch.setattr(
        predictor_module, "AuthenticationPredictor", _SlowDummyEngine
    )

    assert predictor_module._predictor is None
    assert _SlowDummyEngine.construction_count == 0

    predictor_module.get_predictor()

    assert _SlowDummyEngine.construction_count == 1
