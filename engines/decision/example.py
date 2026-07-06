"""
example.py
==========

Runnable demonstration of the refactored package:
  - WeightedVoting fusion combining probability distributions
  - An additional "fraud_ai" recommender voting alongside AI + Policy
  - A HIGH (non-critical) policy priority, so fusion can still win
  - Event hooks
  - Metrics snapshot
  - Ensemble input (List[prediction]) in a second example

Run with:  python -m decision_engine.example
"""

from __future__ import annotations

import json

from . import (
    DecisionConfig,
    DecisionEngine,
    HookRegistry,
    MetricsCollector,
    WeightedVoting,
)


class DummyAuth:
    trust_score = 0.95
    risk_score = 0.12
    confidence = 0.91
    confidence_std = 0.06
    # The network's own predicted distribution — the engine takes the
    # argmax of THIS instead of an invented heuristic.
    decision_probabilities = {
        "ALLOW": 0.82,
        "VOICE_CHALLENGE": 0.11,
        "VOICE_AND_OTP": 0.05,
        "REJECT": 0.02,
    }
    # SHAP-style / integrated-gradient style attributions, preferred over
    # raw attention by ExplanationBuilder.
    attributions = {
        "voice_match": 0.94,
        "device_trust": 0.91,
        "location": 0.20,
        "behavior": 0.40,
    }


class DummyRisk:
    overall_risk = 0.22
    risk_level = "LOW"
    device_known = True
    location_known = False
    voice_match_low = False
    breakdown = {
        "behavior": 0.12,
        "voice": 0.10,
        "transaction": 0.20,
        "device": 0.11,
        "location": 0.30,
    }


class DummyPolicy:
    required_action = "VOICE_CHALLENGE"
    priority = "HIGH"  # not CRITICAL -> participates in fusion, doesn't auto-win
    matched_policy = "MediumRiskPolicy"
    reason = "New location detected for known device."
    rule_trace = [
        {"name": "LargeTransaction", "passed": True},
        {"name": "GeoRule", "passed": False},
        {"name": "VelocityRule", "passed": True},
    ]


def main() -> None:
    hooks = HookRegistry()
    hooks.on("after_fusion", lambda result: print(
        f"[hook] fusion decided {result.action.value} via {result.decision_source}"
    ))

    metrics = MetricsCollector()

    engine = DecisionEngine(
        config=DecisionConfig(),
        strategy=WeightedVoting(),
        metrics=metrics,
        hooks=hooks,
    )

    result = engine.decide(
        authentication=DummyAuth(),
        risk=DummyRisk(),
        policy=DummyPolicy(),
        transaction={"transaction_id": "txn_001", "device_id": "dev_123"},
        # A second model voting alongside AI + Policy. Because the AI and
        # fraud_ai both lean ALLOW while policy leans VOICE_CHALLENGE,
        # weighted fusion can end up siding with the majority weight
        # rather than always deferring to policy.
        additional_recommendations={"fraud_ai": "ALLOW"},
        additional_probabilities={"fraud_ai": {"ALLOW": 0.88, "VOICE_CHALLENGE": 0.12}},
    )

    print(json.dumps(result.to_json(), indent=2, default=str))
    print("\nMetrics snapshot:")
    print(json.dumps(metrics.snapshot(), indent=2))


if __name__ == "__main__":
    main()
