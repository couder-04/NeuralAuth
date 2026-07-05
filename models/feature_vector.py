from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class FeatureVector:
    """
    Numerical feature vector consumed by the
    Adaptive Authentication Network.
    """

    # =====================================================
    # Identity Features
    # =====================================================

    account_age_days: int

    kyc_verified: int

    phone_verified: int

    email_verified: int

    voice_enrolled: int

    # =====================================================
    # Voice Biometrics
    # =====================================================

    speaker_similarity: float

    liveness_score: float

    audio_quality: float

    spoof_probability: float

    # =====================================================
    # Behavioural Features
    # =====================================================

    speech_rate_similarity: float

    pronunciation_similarity: float

    command_familiarity: float

    stress_score: float

    hesitation_score: float

    # =====================================================
    # Vehicle Context
    # =====================================================

    vehicle_speed: float

    engine_running: int

    location_familiarity: float

    time_familiarity: float

    driver_present: int

    seatbelt_fastened: int

    # =====================================================
    # Historical Profile
    # =====================================================

    previous_trust_score: float

    failed_attempts: int

    successful_transactions: int

    fraud_history: int

    # =====================================================
    # Transaction Features
    # =====================================================

    transaction_amount: float

    transaction_category: int

    beneficiary_type: int

    beneficiary_frequency: float

    # =====================================================
    # Intent Features
    # =====================================================

    intent_type: int

    llm_confidence: float

    # =====================================================
    # Risk Features
    # =====================================================

    transaction_risk: float

    # -----------------------------------------------------

    def to_dict(self) -> Dict:

        return asdict(self)