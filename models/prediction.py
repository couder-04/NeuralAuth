from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class Prediction:
    """
    Output of the Adaptive Authentication Network.
    """

    # =====================================================
    # Multi-Task Predictions
    # =====================================================

    trust_score: float
    risk_score: float

    decision: str

    confidence: float

    # =====================================================
    # Explainability
    # =====================================================

    reasons: List[str] = field(default_factory=list)

    # =====================================================
    # Head Probabilities
    # =====================================================

    allow_probability: float = 0.0

    voice_challenge_probability: float = 0.0

    voice_otp_probability: float = 0.0

    reject_probability: float = 0.0

    # =====================================================
    # Metadata
    # =====================================================

    latency_ms: float = 0.0

    model_version: str = "v1.0"

    def to_dict(self) -> Dict:

        return asdict(self)