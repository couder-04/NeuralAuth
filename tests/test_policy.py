"""
tests/test_policy.py

Unit tests for Policy Engine.
"""

from engines.policy_engine import (
    PolicyEngine,
    PolicyInput,
    PolicyAction,
)


def create_input(**kwargs):

    data = dict(

        trust_score=0.95,

        risk_score=0.10,

        confidence=0.95,

        network_decision="ALLOW",

        intent="MONEY_TRANSFER",

        intent_confidence=0.98,

        risk_level="LOW",

        transaction_amount=1000,

        beneficiary_type="KNOWN",

        location_familiarity="FAMILIAR",

        time_familiarity="NORMAL",

        previous_trust_score=0.95,

        failed_attempts=0,

    )

    data.update(kwargs)

    return PolicyInput(**data)


# ---------------------------------------------------------
# Engine Creation
# ---------------------------------------------------------

def test_engine_initialization():

    engine = PolicyEngine()

    assert engine is not None

    assert len(engine.rules) > 0


# ---------------------------------------------------------
# Default Allow
# ---------------------------------------------------------

def test_default_allow():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input()

    )

    assert result.required_action == PolicyAction.ALLOW


# ---------------------------------------------------------
# Reject Critical Risk
# ---------------------------------------------------------

def test_reject_critical_risk():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            risk_level="CRITICAL"

        )

    )

    assert result.required_action == PolicyAction.REJECT


# ---------------------------------------------------------
# Reject Low Trust
# ---------------------------------------------------------

def test_reject_low_trust():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            trust_score=0.05

        )

    )

    assert result.required_action == PolicyAction.REJECT


# ---------------------------------------------------------
# Manual Review
# ---------------------------------------------------------

def test_manual_review_failed_attempts():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            failed_attempts=5

        )

    )

    assert result.required_action == PolicyAction.MANUAL_REVIEW


# ---------------------------------------------------------
# Voice + OTP
# ---------------------------------------------------------

def test_voice_and_otp():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            risk_level="HIGH"

        )

    )

    assert result.required_action == PolicyAction.VOICE_AND_OTP


# ---------------------------------------------------------
# OTP
# ---------------------------------------------------------

def test_large_transaction():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            transaction_amount=500000

        )

    )

    assert result.required_action == PolicyAction.OTP


# ---------------------------------------------------------
# New Beneficiary
# ---------------------------------------------------------

def test_new_beneficiary():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            beneficiary_type="NEW"

        )

    )

    assert result.required_action == PolicyAction.VOICE_AND_OTP


# ---------------------------------------------------------
# Low Intent Confidence
# ---------------------------------------------------------

def test_low_intent_confidence():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            intent_confidence=0.20

        )

    )

    assert result.required_action == PolicyAction.VOICE_CHALLENGE


# ---------------------------------------------------------
# Odd Hour
# ---------------------------------------------------------

def test_odd_hour():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            time_familiarity="ODD_HOUR"

        )

    )

    assert result.required_action == PolicyAction.VOICE_CHALLENGE


# ---------------------------------------------------------
# Unfamiliar Location
# ---------------------------------------------------------

def test_unfamiliar_location():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            location_familiarity="UNFAMILIAR"

        )

    )

    assert result.required_action == PolicyAction.OTP


# ---------------------------------------------------------
# Rule Priority
# ---------------------------------------------------------

def test_priority_resolution():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            risk_level="CRITICAL",

            trust_score=0.01,

            failed_attempts=8,

        )

    )

    assert result.required_action == PolicyAction.REJECT


# ---------------------------------------------------------
# Audit Log
# ---------------------------------------------------------

def test_audit_log():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            risk_level="HIGH"

        )

    )

    audit = result.audit_message

    assert isinstance(audit, dict)

    assert "timestamp" in audit

    assert "matched_policy" in audit

    assert "action" in audit


# ---------------------------------------------------------
# Export Policy
# ---------------------------------------------------------

def test_export_policy():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input()

    )

    exported = engine.export_policy(result)

    assert isinstance(exported, dict)

    assert exported["required_action"] == "ALLOW"


# ---------------------------------------------------------
# Override Flag
# ---------------------------------------------------------

def test_override_network_decision():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            network_decision="ALLOW",

            risk_level="HIGH",

        )

    )

    assert result.override_network_decision is True


# ---------------------------------------------------------
# Rule Matching
# ---------------------------------------------------------

def test_match_rule():

    engine = PolicyEngine()

    ctx = create_input(

        risk_level="HIGH"

    ).as_dict()

    matches = engine.match_rule(ctx)

    assert len(matches) > 0


# ---------------------------------------------------------
# No Crash On Unknown Values
# ---------------------------------------------------------

def test_unknown_values():

    engine = PolicyEngine()

    result = engine.evaluate(

        create_input(

            beneficiary_type="UNKNOWN",

            location_familiarity="UNKNOWN",

            time_familiarity="UNKNOWN",

        )

    )

    assert result is not None