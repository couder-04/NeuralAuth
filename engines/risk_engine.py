"""
risk_engine.py
==============

Input:
    Authentication Network Output

Output:
    Risk Assessment

The Risk Engine does NOT compute risk itself.

Risk prediction is produced by the Authentication Network.

This module simply standardizes that prediction into a
typed object for downstream policy evaluation.
"""

from dataclasses import dataclass
from typing import Dict


# ==========================================================
# Risk Result
# ==========================================================

@dataclass
class RiskResult:

    risk_score: float

    risk_level: str

    confidence: float

    def to_dict(self) -> Dict:

        return {

            "risk_score": self.risk_score,

            "risk_level": self.risk_level,

            "confidence": self.confidence

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
    intent,
    features,
) -> RiskResult:

        risk_score = authentication.risk_score.squeeze().item()
        confidence = authentication.confidence.squeeze().item()

        return self.process(
            risk_score=risk_score,
            confidence=confidence,
        )

    @staticmethod
    def process(
        risk_score: float,
        confidence: float
    ) -> RiskResult:

        return RiskResult(

            risk_score=round(risk_score, 2),

            risk_level=RiskEngine.classify(risk_score),

            confidence=round(confidence, 4)

        )


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    result = RiskEngine.process(

        risk_score=72.4,

        confidence=0.984

    )

    print(result.to_dict())