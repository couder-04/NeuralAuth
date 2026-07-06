"""
tests/test_dashboard_feature_vector.py

Regression test for the reported bug: the dashboard's "Feature Vector"
card showed far fewer than all 31 FeatureVector fields.

Root cause (see engines/decision/audit.py / decision_engine.py /
api/server.py):
    1. The full FeatureVector was computed in api/server.py but never
       passed to DecisionEngine.decide(), so it never reached
       audit_log at all.
    2. dashboard.py's `find_dict_by_keywords(audit_log, ("feature_vector",
       "features"))` did a substring search over audit_log's *keys* and
       happened to match `top_features` (a top-5 model-attribution dict
       from ExplanationBuilder.top_contributors), because "features" is
       a substring of "top_features" -- so the dashboard displayed that
       instead, capped at 5 entries.

Fix: DecisionEngine.decide() now accepts `features` and includes the
full dict under `audit_log["feature_vector"]`; the old `top_features`
key was renamed to `top_attributions` so it can no longer collide with
the keyword search.
"""

from dashboard import find_dict_by_keywords

from engines.authentication_network import AuthenticationResult
from engines.decision import DecisionEngine
from engines.feature_extractor import FeatureExtractor
from engines.policy_engine import PolicyEngine, PolicyInput
from engines.risk_engine import RiskEngine


def _build_real_audit_log():
    auth = AuthenticationResult(
        trust_score=0.9,
        risk_score=0.2,
        confidence=0.95,
        recommended_action="ALLOW",
        decision_probabilities={
            "ALLOW": 0.9,
            "VOICE_CHALLENGE": 0.05,
            "VOICE_AND_OTP": 0.03,
            "REJECT": 0.02,
        },
        attributions={"transaction_risk": 0.2, "location_familiarity": 0.1},
        confidence_std=None,
        embedding=None,
        model_version="test-v1",
        latency_ms=1.0,
    )

    features = FeatureExtractor.extract(
        {
            "identity": {"account_age_days": 365, "kyc_verified": True},
            "vehicle": {"location_familiarity": 0.9, "time_familiarity": 0.9},
            "history": {"failed_attempts": 0},
            "transaction": {"amount": 2500},
        }
    )

    risk = RiskEngine().evaluate(authentication=auth, features=features)
    policy = PolicyEngine().evaluate(
        PolicyInput(
            trust_score=auth.trust_score,
            risk_score=auth.risk_score,
            confidence=auth.confidence,
            network_decision=auth.recommended_action,
            intent="MONEY_TRANSFER",
            intent_confidence=0.9,
            risk_level=risk.risk_level,
            transaction_amount=2500,
            beneficiary_type="KNOWN",
        )
    )
    decision = DecisionEngine().decide(
        authentication=auth,
        risk=risk,
        policy=policy,
        transaction={"user_id": "test-user"},
        features=features,
    )
    return decision.audit_log, features


def test_dashboard_feature_vector_card_resolves_to_all_31_fields():
    audit_log, features = _build_real_audit_log()

    resolved = find_dict_by_keywords(audit_log, ("feature_vector", "features"))

    assert resolved is not None
    assert len(resolved) == 31
    assert resolved == features.to_dict()


def test_dashboard_feature_vector_card_no_longer_matches_top_attributions():
    audit_log, _ = _build_real_audit_log()

    resolved = find_dict_by_keywords(audit_log, ("feature_vector", "features"))

    # Must not accidentally resolve to the (much smaller) attribution dict.
    assert resolved != audit_log["top_attributions"]
    assert len(resolved) > len(audit_log["top_attributions"])
