"""
feature_extractor.py
====================

Input:
    Raw JSON

Output:
    FeatureVector

Responsibilities
----------------
1. Extract features from raw JSON.
2. Validate missing fields.
3. Return a strongly typed FeatureVector.

This module performs NO machine learning,
NO scoring,
NO normalization,
NO tensor conversion.

Those belong to the training/inference pipeline.
"""

from dataclasses import asdict
from typing import Any, Dict

from models.feature_vector import FeatureVector


# =====================================================
# Categorical Encoders
# =====================================================

TRANSACTION_CATEGORY_MAP = {
    "TRANSFER": 0,
    "UPI": 1,
    "BILL": 2,
    "SHOPPING": 3,
    "ATM": 4,
    "OTHER": 5,
}

BENEFICIARY_TYPE_MAP = {
    "SAVED": 0,
    "NEW": 1,
    "SELF": 2,
    "MERCHANT": 3,
    "OTHER": 4,
}

INTENT_TYPE_MAP = {
    "UNKNOWN": 0,
    "BALANCE_INQUIRY": 1,
    "MONEY_TRANSFER": 2,
    "BILL_PAYMENT": 3,
    "TRANSACTION_HISTORY": 4,
}


def encode(value, mapping):
    """
    Supports both encoded datasets and API requests.

    Examples
    --------
    "TRANSFER" -> 0
    "transfer" -> 0
    0 -> 0
    """

    if isinstance(value, int):
        return value

    return mapping.get(
        str(value).upper(),
        0,
    )


# =====================================================
# Feature Extractor
# =====================================================

class FeatureExtractor:

    @staticmethod
    def extract(data: Dict[str, Any]) -> FeatureVector:

        identity = data.get("identity", {})
        biometric = data.get("biometric", {})
        behavior = data.get("behavior", {})
        vehicle = data.get("vehicle", {})
        history = data.get("history", {})
        transaction = data.get("transaction", {})

        return FeatureVector(

            # =====================================================
            # Identity
            # =====================================================

            account_age_days=int(
                identity.get("account_age_days", 0)
            ),

            kyc_verified=int(
                identity.get("kyc_verified", False)
            ),

            phone_verified=int(
                identity.get("phone_verified", False)
            ),

            email_verified=int(
                identity.get("email_verified", False)
            ),

            voice_enrolled=int(
                identity.get("voice_enrolled", False)
            ),

            # =====================================================
            # Voice Biometrics
            # =====================================================

            speaker_similarity=float(
                biometric.get("speaker_similarity", 0.0)
            ),

            liveness_score=float(
                biometric.get("liveness_score", 0.0)
            ),

            audio_quality=float(
                biometric.get("audio_quality", 0.0)
            ),

            spoof_probability=float(
                biometric.get("spoof_probability", 0.0)
            ),

            # =====================================================
            # Behavioural Features
            # =====================================================

            speech_rate_similarity=float(
                behavior.get("speech_rate_similarity", 0.0)
            ),

            pronunciation_similarity=float(
                behavior.get("pronunciation_similarity", 0.0)
            ),

            command_familiarity=float(
                behavior.get("command_familiarity", 0.0)
            ),

            stress_score=float(
                behavior.get("stress_score", 0.0)
            ),

            hesitation_score=float(
                behavior.get("hesitation_score", 0.0)
            ),

            # =====================================================
            # Vehicle Context
            # =====================================================

            vehicle_speed=float(
                vehicle.get("vehicle_speed", 0.0)
            ),

            engine_running=int(
                vehicle.get("engine_running", False)
            ),

            location_familiarity=float(
                vehicle.get("location_familiarity", 0.0)
            ),

            time_familiarity=float(
                vehicle.get("time_familiarity", 0.0)
            ),

            driver_present=int(
                vehicle.get("driver_present", False)
            ),

            seatbelt_fastened=int(
                vehicle.get("seatbelt_fastened", False)
            ),

            # =====================================================
            # Historical Profile
            # =====================================================

            previous_trust_score=float(
                history.get("previous_trust_score", 1.0)
            ),

            failed_attempts=int(
                history.get("failed_attempts", 0)
            ),

            successful_transactions=int(
                history.get("successful_transactions", 0)
            ),

            fraud_history=int(
                history.get("fraud_history", False)
            ),

            # =====================================================
            # Transaction Features
            # =====================================================

            transaction_amount=float(
                transaction.get("amount", 0.0)
            ),

            transaction_category=encode(
                transaction.get("category", "OTHER"),
                TRANSACTION_CATEGORY_MAP,
            ),

            beneficiary_type=encode(
                transaction.get("beneficiary_type", "OTHER"),
                BENEFICIARY_TYPE_MAP,
            ),

            beneficiary_frequency=float(
                transaction.get("beneficiary_frequency", 0.0)
            ),

            # =====================================================
            # Intent Features
            # =====================================================

            intent_type=encode(
                transaction.get("intent", "UNKNOWN"),
                INTENT_TYPE_MAP,
            ),

            llm_confidence=float(
                transaction.get("llm_confidence", 0.0)
            ),

            # =====================================================
            # Risk Features
            # =====================================================

            transaction_risk=float(
                transaction.get("transaction_risk", 0.0)
            ),
        )

    @staticmethod
    def to_dict(feature_vector: FeatureVector) -> Dict:

        return asdict(feature_vector)