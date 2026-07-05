"""
decision_engine.py
==================

Final orchestration layer of the Transaction Authentication Engine.

Responsibilities
----------------
1. Merge outputs from all previous engines.
2. Produce ONE final decision.
3. Generate API-friendly response.
4. Produce audit information.

This module performs NO machine learning.
It simply combines outputs from:

- Authentication Network
- Intent Engine
- Risk Engine
- Policy Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


# ============================================================
# Final Actions
# ============================================================

class DecisionAction(str, Enum):

    ALLOW = "ALLOW"

    VOICE_CHALLENGE = "VOICE_CHALLENGE"

    OTP = "OTP"

    VOICE_AND_OTP = "VOICE_AND_OTP"

    MANUAL_REVIEW = "MANUAL_REVIEW"

    REJECT = "REJECT"


# ============================================================
# Final Output
# ============================================================

@dataclass
class DecisionResult:

    status: str

    action: DecisionAction

    transaction_allowed: bool

    authentication_required: bool

    voice_required: bool

    otp_required: bool

    manual_review: bool

    message: str

    reason: str

    audit_log: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Decision Engine
# ============================================================

class DecisionEngine:

    """
    Converts the outputs of every engine into a final response.
    """

    # --------------------------------------------------------

    def decide(

        self,

        authentication,

        risk,

        policy,

        intent=None,

        transaction: Optional[Dict[str, Any]] = None,

    ) -> DecisionResult:

        """
        Parameters
        ----------

        authentication
            Output of Authentication Network.

        risk
            Output of Risk Engine.

        policy
            Output of Policy Engine.

        intent
            Optional output from Intent Engine.

        transaction
            Original transaction.
        """

        action = DecisionAction(policy.required_action.value)

        message = self._message(action)

        allowed = action == DecisionAction.ALLOW

        voice = action in (

            DecisionAction.VOICE_CHALLENGE,

            DecisionAction.VOICE_AND_OTP,

        )

        otp = action in (

            DecisionAction.OTP,

            DecisionAction.VOICE_AND_OTP,

        )

        manual = action == DecisionAction.MANUAL_REVIEW

        auth_required = voice or otp

        audit = self._audit(

            authentication,

            risk,

            policy,

            intent,

            transaction,

            action,

        )

        return DecisionResult(

            status="SUCCESS",

            action=action,

            transaction_allowed=allowed,

            authentication_required=auth_required,

            voice_required=voice,

            otp_required=otp,

            manual_review=manual,

            message=message,

            reason=policy.reason,

            audit_log=audit,

        )

    # --------------------------------------------------------
    @staticmethod
    def _to_python(value):
        if value is None:
            return None

        try:
            import torch

            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.item()
                return value.detach().cpu().tolist()
        except ImportError:
            pass

        return value
    @staticmethod
    def _message(action: DecisionAction) -> str:

        messages = {

            DecisionAction.ALLOW:
                "Transaction Approved.",

            DecisionAction.VOICE_CHALLENGE:
                "Voice verification required.",

            DecisionAction.OTP:
                "OTP verification required.",

            DecisionAction.VOICE_AND_OTP:
                "Voice verification and OTP required.",

            DecisionAction.MANUAL_REVIEW:
                "Transaction sent for manual review.",

            DecisionAction.REJECT:
                "Transaction rejected.",

        }

        return messages[action]

    # --------------------------------------------------------

    @staticmethod
    def _audit(

        authentication,

        risk,

        policy,

        intent,

        transaction,

        action,

    ) -> Dict[str, Any]:

        audit = {

            "timestamp":
                datetime.utcnow().isoformat(),

            "decision":
                action.value,

            "trust_score":
                DecisionEngine._to_python(
                    getattr(authentication, "trust_score", None)
                ),

            "risk_score":
                DecisionEngine._to_python(
                    getattr(authentication, "risk_score", None)
                ),

            "confidence":
                DecisionEngine._to_python(
                    getattr(authentication, "confidence", None)
                ),

            "risk_level":
                getattr(risk, "risk_level", None),

            "overall_risk":
                DecisionEngine._to_python(
                    getattr(risk, "overall_risk", None)
                ),

            "policy":
                getattr(policy, "policy_name", None),

            "reason":
                getattr(policy, "reason", None),

        }

        if intent is not None:

            audit["intent"] = getattr(intent, "intent", None)

            audit["intent_confidence"] = getattr(

                intent,

                "confidence",

                None,

            )

        if transaction is not None:

            audit["transaction"] = transaction

        return audit


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    class DummyAuth:

        trust_score = 0.95

        risk_score = 0.12

        confidence = 0.98


    class DummyRisk:

        overall_risk = 0.18

        risk_level = "LOW"


    class DummyPolicy:

        required_action = "ALLOW"

        matched_policy = "DefaultPolicy"

        reason = "Low risk transaction."


    engine = DecisionEngine()

    result = engine.decide(

        authentication=DummyAuth(),

        risk=DummyRisk(),

        policy=DummyPolicy(),

    )

    print(result)