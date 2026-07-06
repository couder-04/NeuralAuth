"""
risk_engine.py
==============

Input:
    AuthenticationResult (from the Authentication Network), plus
    optionally the Intent Engine's Transaction and the raw FeatureVector.

Output:
    RiskResult

The Risk Engine does NOT compute risk itself.

The core risk number is produced by the Authentication Network. This
module standardizes that prediction into a typed object for downstream
policy evaluation, and packages the signals already present on the
authentication result / feature vector into an auditable breakdown
instead of silently discarding them.

Downstream contract (see engines/decision/decision_engine.py and
engines/decision/audit.py, which read these via `getattr`):

    overall_risk        float  0..1, the authoritative risk number
    risk_level           str    LOW | MEDIUM | HIGH | CRITICAL
    confidence           float
    breakdown            Dict[str, float]  named risk components

Deliberately NOT part of this contract: any kind of recommended action.
Risk assesses; it doesn't decide. `overall_risk` flows into
RiskWeightedFusion (engines/decision/fusion.py) as continuous evidence
that shifts the fused probability distribution -- it is never converted
into a discrete action here, so there is only one place (Decision
Fusion) that turns risk into a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# ==========================================================
# Risk Result
# ==========================================================


@dataclass
class RiskResult:

    overall_risk: float

    risk_level: str

    confidence: float

    breakdown: Dict[str, float] = field(default_factory=dict)

    # Kept for backward compatibility with callers/tests that still read
    # `risk_score` directly -- always equal to `overall_risk`.
    @property
    def risk_score(self) -> float:
        return self.overall_risk

    def to_dict(self) -> Dict:

        return {

            "overall_risk": self.overall_risk,

            "risk_score": self.overall_risk,

            "risk_level": self.risk_level,

            "confidence": self.confidence,

            "breakdown": self.breakdown,

        }


# ==========================================================
# Risk Engine
# ==========================================================

class RiskEngine:

    LOW_THRESHOLD = 0.30

    MEDIUM_THRESHOLD = 0.60

    HIGH_THRESHOLD = 0.85

    @staticmethod
    def classify(score: float) -> str:

        if score < RiskEngine.LOW_THRESHOLD:
            return "LOW"

        if score < RiskEngine.MEDIUM_THRESHOLD:
            return "MEDIUM"

        if score < RiskEngine.HIGH_THRESHOLD:
            return "HIGH"

        return "CRITICAL"

    def evaluate(
        self,
        authentication,
        intent=None,
        features=None,
    ) -> RiskResult:
        """
        Parameters
        ----------
        authentication
            An `AuthenticationResult` (plain-Python, see
            engines/authentication_network.py) -- or, for backward
            compatibility, a raw tensor-based `Prediction`.
        intent
            Optional Intent Engine `Transaction`.
        features
            Optional `FeatureVector` (or dict) used to build the risk
            `breakdown`. Never required for the top-level risk score.
        """

        risk_score = self._extract(authentication, "risk_score")
        confidence = self._extract(authentication, "confidence")

        breakdown = self.build_breakdown(features, intent)

        return self.process(
            risk_score=risk_score,
            confidence=confidence,
            breakdown=breakdown,
        )

    @staticmethod
    def _extract(authentication, field_name: str) -> float:
        """Read a scalar field off either a plain-Python AuthenticationResult
        or a raw batched tensor Prediction (kept only for callers that
        haven't migrated to AuthenticationResult yet)."""
        value = getattr(authentication, field_name)
        squeeze = getattr(value, "squeeze", None)
        if callable(squeeze):
            # torch.Tensor path
            return float(squeeze().item())
        return float(value)

    @staticmethod
    def build_breakdown(features, intent=None) -> Dict[str, float]:
        """Package the signals already computed upstream (voice biometrics,
        location/time familiarity, transaction risk, ...) into a named
        breakdown instead of collapsing everything into a single opaque
        number. Best-effort: missing signals are simply omitted.
        """
        if features is None:
            return {}

        get = (
            features.get
            if isinstance(features, dict)
            else lambda name, default=None: getattr(features, name, default)
        )

        breakdown: Dict[str, float] = {}

        speaker_similarity = get("speaker_similarity")
        spoof_probability = get("spoof_probability")
        if speaker_similarity is not None or spoof_probability is not None:
            voice_risk = 0.0
            if speaker_similarity is not None:
                voice_risk = max(voice_risk, 1.0 - float(speaker_similarity))
            if spoof_probability is not None:
                voice_risk = max(voice_risk, float(spoof_probability))
            breakdown["voice_risk"] = round(voice_risk, 4)

        stress_score = get("stress_score")
        hesitation_score = get("hesitation_score")
        if stress_score is not None or hesitation_score is not None:
            behavior_risk = max(
                float(stress_score or 0.0),
                float(hesitation_score or 0.0),
            )
            breakdown["behavior_risk"] = round(behavior_risk, 4)

        location_familiarity = get("location_familiarity")
        if location_familiarity is not None:
            breakdown["location_risk"] = round(1.0 - float(location_familiarity), 4)

        time_familiarity = get("time_familiarity")
        if time_familiarity is not None:
            breakdown["device_risk"] = round(1.0 - float(time_familiarity), 4)

        transaction_risk = get("transaction_risk")
        if transaction_risk is not None:
            breakdown["transaction_risk"] = round(float(transaction_risk), 4)

        return breakdown

    @staticmethod
    def process(
        risk_score: float,
        confidence: float,
        breakdown: Optional[Dict[str, float]] = None,
    ) -> RiskResult:

        risk_level = RiskEngine.classify(risk_score)

        return RiskResult(

            overall_risk=round(risk_score, 2),

            risk_level=risk_level,

            confidence=round(confidence, 4),

            breakdown=breakdown or {},

            recommended_action=RiskEngine.recommended_action_for(risk_level),

        )


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    result = RiskEngine.process(

        risk_score=0.724,

        confidence=0.984,

    )

    print(result.to_dict())