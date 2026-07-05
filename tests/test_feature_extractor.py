"""
tests/test_feature_extractor.py

Unit tests for FeatureExtractor.
"""

from engines.feature_extractor import FeatureExtractor
from models.feature_vector import FeatureVector


# ---------------------------------------------------------
# Sample Request
# ---------------------------------------------------------

def sample_request():

    return {

        "identity": {
            "account_age_days": 365,
            "kyc_verified": True,
            "phone_verified": True,
            "email_verified": True,
            "voice_enrolled": True,
        },

        "biometric": {
            "speaker_similarity": 0.96,
            "liveness_score": 0.98,
            "audio_quality": 0.95,
            "spoof_probability": 0.02,
        },

        "behavior": {
            "speech_rate_similarity": 0.90,
            "pronunciation_similarity": 0.92,
            "command_familiarity": 0.88,
            "stress_score": 0.10,
            "hesitation_score": 0.05,
        },

        "vehicle": {
            "driver_present": True,
            "seatbelt_fastened": True,
            "engine_running": True,
            "vehicle_speed": 45,
            "location_familiarity": 0.92,
            "time_familiarity": 0.81,
        },

        "history": {
            "failed_attempts": 1,
            "previous_trust_score": 0.93,
            "successful_transactions": 120,
            "fraud_history": False,
        },

        "transaction": {
            "amount": 25000,
            "category": 1,
            "beneficiary_type": 0,
            "beneficiary_frequency": 0.85,
            "intent": 2,
            "llm_confidence": 0.97,
            "transaction_risk": 0.18,
        },
    }


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

def test_feature_extraction():

    fv = FeatureExtractor.extract(sample_request())

    assert isinstance(fv, FeatureVector)


# ---------------------------------------------------------
# Identity Features
# ---------------------------------------------------------

def test_identity_features():

    fv = FeatureExtractor.extract(sample_request())

    assert fv.account_age_days == 365
    assert fv.kyc_verified == 1
    assert fv.phone_verified == 1
    assert fv.email_verified == 1
    assert fv.voice_enrolled == 1


# ---------------------------------------------------------
# Biometric Features
# ---------------------------------------------------------

def test_biometric_features():

    fv = FeatureExtractor.extract(sample_request())

    assert fv.speaker_similarity == 0.96
    assert fv.liveness_score == 0.98
    assert fv.audio_quality == 0.95
    assert fv.spoof_probability == 0.02


# ---------------------------------------------------------
# Behaviour Features
# ---------------------------------------------------------

def test_behavior_features():

    fv = FeatureExtractor.extract(sample_request())

    assert fv.speech_rate_similarity == 0.90
    assert fv.pronunciation_similarity == 0.92
    assert fv.command_familiarity == 0.88
    assert fv.stress_score == 0.10
    assert fv.hesitation_score == 0.05


# ---------------------------------------------------------
# Vehicle Features
# ---------------------------------------------------------

def test_vehicle_features():

    fv = FeatureExtractor.extract(sample_request())

    assert fv.driver_present == 1
    assert fv.seatbelt_fastened == 1
    assert fv.engine_running == 1
    assert fv.vehicle_speed == 45
    assert fv.location_familiarity == 0.92
    assert fv.time_familiarity == 0.81


# ---------------------------------------------------------
# History Features
# ---------------------------------------------------------

def test_history_features():

    fv = FeatureExtractor.extract(sample_request())

    assert fv.previous_trust_score == 0.93
    assert fv.failed_attempts == 1
    assert fv.successful_transactions == 120
    assert fv.fraud_history == 0


# ---------------------------------------------------------
# Transaction Features
# ---------------------------------------------------------

def test_transaction_features():

    fv = FeatureExtractor.extract(sample_request())

    assert fv.transaction_amount == 25000
    assert fv.transaction_category == 1
    assert fv.beneficiary_type == 0
    assert fv.beneficiary_frequency == 0.85
    assert fv.intent_type == 2
    assert fv.llm_confidence == 0.97
    assert fv.transaction_risk == 0.18


# ---------------------------------------------------------
# Missing Fields
# ---------------------------------------------------------

def test_missing_fields():

    fv = FeatureExtractor.extract({})

    assert isinstance(fv, FeatureVector)

    assert fv.account_age_days == 0
    assert fv.transaction_amount == 0
    assert fv.failed_attempts == 0
    assert fv.previous_trust_score == 1.0
    assert fv.successful_transactions == 0
    assert fv.fraud_history == 0
    assert fv.transaction_risk == 0.0


# ---------------------------------------------------------
# Partial Request
# ---------------------------------------------------------

def test_partial_request():

    request = {
        "identity": {
            "kyc_verified": True,
        }
    }

    fv = FeatureExtractor.extract(request)

    assert fv.kyc_verified == 1
    assert fv.account_age_days == 0
    assert fv.transaction_amount == 0.0


# ---------------------------------------------------------
# Numeric Types
# ---------------------------------------------------------

def test_numeric_types():

    fv = FeatureExtractor.extract(sample_request())

    assert isinstance(fv.account_age_days, int)
    assert isinstance(fv.kyc_verified, int)
    assert isinstance(fv.transaction_amount, float)
    assert isinstance(fv.failed_attempts, int)
    assert isinstance(fv.transaction_risk, float)
    assert isinstance(fv.llm_confidence, float)


# ---------------------------------------------------------
# to_dict()
# ---------------------------------------------------------

def test_to_dict():

    fv = FeatureExtractor.extract(sample_request())

    d = fv.to_dict()

    assert isinstance(d, dict)

    assert "transaction_amount" in d
    assert "speaker_similarity" in d
    assert "previous_trust_score" in d
    assert "transaction_risk" in d
    assert "llm_confidence" in d


# ---------------------------------------------------------
# Boolean Conversion
# ---------------------------------------------------------

def test_boolean_conversion():

    request = sample_request()

    request["identity"]["kyc_verified"] = False
    request["vehicle"]["driver_present"] = False

    fv = FeatureExtractor.extract(request)

    assert fv.kyc_verified == 0
    assert fv.driver_present == 0