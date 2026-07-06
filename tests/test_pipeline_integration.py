"""
tests/test_pipeline_integration.py

End-to-end integration tests for the M0-M10 pipeline:

    AuthenticationResult -> Risk Engine -> Policy Engine -> Decision Engine

Unlike tests/test_api.py (which mocks every engine to test the FastAPI
wiring), these tests run the REAL Risk / Policy / Decision engines
against a real `AuthenticationResult`, to verify the engine *contracts*
line up end-to-end without adapters or fallback heuristics (this is
what the architecture review's "Integration" and "Engine Contracts"
sections were about).
"""

from engines.authentication_network import AuthenticationResult
from engines.decision import DecisionEngine
from engines.decision.types import DecisionAction
from engines.policy_engine import PolicyEngine, PolicyInput
from engines.risk_engine import RiskEngine


def make_auth_result(**overrides) -> AuthenticationResult:
    defaults = dict(
        trust_score=0.92,
        risk_score=0.12,
        confidence=0.95,
        recommended_action="ALLOW",
        decision_probabilities={
            "ALLOW": 0.92,
            "VOICE_CHALLENGE": 0.05,
            "VOICE_AND_OTP": 0.02,
            "REJECT": 0.01,
        },
        attributions={"transaction_risk": 0.1, "location_familiarity": 0.05},
        confidence_std=None,
        embedding=None,
        model_version="test-v1",
        latency_ms=1.0,
    )
    defaults.update(overrides)
    return AuthenticationResult(**defaults)


def run_pipeline(auth: AuthenticationResult, features=None, **policy_overrides):
    risk = RiskEngine().evaluate(authentication=auth, features=features)

    policy_input = PolicyInput(
        trust_score=auth.trust_score,
        risk_score=auth.risk_score,
        confidence=auth.confidence,
        network_decision=auth.recommended_action,
        intent="MONEY_TRANSFER",
        intent_confidence=0.9,
        risk_level=risk.risk_level,
        transaction_amount=policy_overrides.pop("transaction_amount", 1000.0),
        beneficiary_type=policy_overrides.pop("beneficiary_type", "KNOWN"),
        **policy_overrides,
    )
    policy = PolicyEngine().evaluate(policy_input)

    decision = DecisionEngine().decide(
        authentication=auth,
        risk=risk,
        policy=policy,
        transaction={"user_id": "test-user"},
    )
    return risk, policy, decision


# ---------------------------------------------------------------------------
# Contract fields exist end-to-end (no adapters / getattr(..., None) misses)
# ---------------------------------------------------------------------------

def test_authentication_result_exposes_decision_engine_contract():
    auth = make_auth_result()
    assert isinstance(auth.decision_probabilities, dict)
    assert auth.recommended_action in auth.decision_probabilities
    assert isinstance(auth.attributions, dict)


def test_risk_result_exposes_overall_risk_and_breakdown():
    auth = make_auth_result(risk_score=0.75)
    risk = RiskEngine().evaluate(
        authentication=auth,
        features={"transaction_risk": 0.8, "location_familiarity": 0.9},
    )
    assert risk.overall_risk == 0.75
    assert risk.risk_level == "HIGH"
    assert "transaction_risk" in risk.breakdown
    assert risk.recommended_action == "VOICE_AND_OTP"


def test_policy_result_exposes_priority_and_rule_trace():
    auth = make_auth_result(risk_score=0.9)
    risk, policy, _ = run_pipeline(auth)
    assert policy.priority in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert isinstance(policy.rule_trace, list)
    assert isinstance(policy.matched_rules, list)
    assert 0.0 <= policy.policy_score <= 1.0


# ---------------------------------------------------------------------------
# Fusion consumes real upstream outputs, not fallback heuristics
# ---------------------------------------------------------------------------

def test_low_risk_allows_transaction():
    auth = make_auth_result()
    risk, policy, decision = run_pipeline(auth)
    assert decision.action == DecisionAction.ALLOW
    assert decision.transaction_allowed is True


def test_critical_risk_forces_reject():
    auth = make_auth_result(
        risk_score=0.95,
        trust_score=0.5,
        recommended_action="REJECT",
        decision_probabilities={
            "ALLOW": 0.01,
            "VOICE_CHALLENGE": 0.02,
            "VOICE_AND_OTP": 0.07,
            "REJECT": 0.90,
        },
    )
    risk, policy, decision = run_pipeline(auth)
    assert risk.risk_level == "CRITICAL"
    # A CRITICAL policy priority must unconditionally win fusion.
    assert policy.priority == "CRITICAL"
    assert decision.action == DecisionAction.REJECT
    assert decision.transaction_allowed is False


def test_new_beneficiary_vote_reaches_fusion():
    auth = make_auth_result(risk_score=0.4, trust_score=0.6)
    risk, policy, decision = run_pipeline(
        auth,
        transaction_amount=250000,
        beneficiary_type="NEW",
    )
    # The Policy Engine must flag this as requiring strong authentication...
    assert policy.required_action.value == "VOICE_AND_OTP"
    # ...and that vote must actually reach the fusion stage (not be dropped
    # by an adapter/contract mismatch), even if a non-CRITICAL policy vote
    # doesn't unconditionally win against a confident AI recommendation.
    votes = decision.audit_log["decision_trace"]["fusion"]["votes"]
    assert votes["policy_recommendation"] == "VOICE_AND_OTP"


def test_audit_log_carries_full_decision_trace():
    auth = make_auth_result(risk_score=0.7)
    risk, policy, decision = run_pipeline(auth)
    trace = decision.audit_log["decision_trace"]
    assert trace["risk"]["overall_risk"] == risk.overall_risk
    assert trace["policy"]["priority"] == policy.priority
    assert trace["policy"]["matched_policy"] == policy.matched_policy
    assert trace["authentication"]["decision_probabilities"] == auth.decision_probabilities
