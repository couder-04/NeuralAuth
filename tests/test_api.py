"""
tests/test_api.py

Integration tests for FastAPI server.
"""

from fastapi.testclient import TestClient

from api.server import app


client = TestClient(app)


# ==========================================================
# Health Endpoints
# ==========================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"

    assert "service" in body


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


# ==========================================================
# Authentication Endpoint
# ==========================================================

def test_authenticate(monkeypatch):

    from api import server

    # --------------------------------------------------
    # Fake Feature Vector
    # --------------------------------------------------

    class DummyFeatureVector:

        def to_dict(self):
            return {}

    monkeypatch.setattr(
        server.FeatureExtractor,
        "extract",
        lambda data: DummyFeatureVector(),
    )

    monkeypatch.setattr(
        server.FeatureExtractor,
        "to_dict",
        lambda fv: {
            "x1": 0.1,
            "x2": 0.2,
            "x3": 0.3,
        },
    )

    # --------------------------------------------------
    # Fake Authentication Model
    # --------------------------------------------------

    monkeypatch.setattr(
        server,
        "get_auth_model",
        lambda: object(),
    )

    # --------------------------------------------------
    # Fake Authentication Prediction
    # --------------------------------------------------

    class DummyPrediction:

        trust_score = 0.95
        risk_score = 0.10
        confidence = 0.98
        decision = "ALLOW"

    monkeypatch.setattr(
        server,
        "predict",
        lambda *args, **kwargs: DummyPrediction(),
    )

    # --------------------------------------------------
    # Fake Intent Engine
    # --------------------------------------------------

    class DummyTransaction:

        intent = "MONEY_TRANSFER"
        amount = 5000
        currency = "INR"
        beneficiary = "Rahul"
        beneficiary_type = "SAVED"
        transaction_category = "P2P_TRANSFER"
        purpose = "PERSONAL_TRANSFER"
        confidence = 0.97

    class DummyParsed:

        transaction = DummyTransaction()

    class DummyIntentEngine:

        def parse(self, transcript):
            return DummyParsed()

    monkeypatch.setattr(
        server,
        "get_intent_engine",
        lambda: DummyIntentEngine(),
    )

    # --------------------------------------------------
    # Fake Risk Engine
    # --------------------------------------------------

    class DummyRisk:

        level = "LOW"
        score = 0.10

    class DummyRiskEngine:

        def evaluate(self, **kwargs):
            return DummyRisk()

    monkeypatch.setattr(
        server,
        "get_risk_engine",
        lambda: DummyRiskEngine(),
    )

    # --------------------------------------------------
    # Fake Policy Engine
    # --------------------------------------------------

    class DummyPolicy:

        required_action = "ALLOW"

    class DummyPolicyEngine:

        def evaluate(self, **kwargs):
            return DummyPolicy()

    monkeypatch.setattr(
        server,
        "get_policy_engine",
        lambda: DummyPolicyEngine(),
    )

    # --------------------------------------------------
    # Fake Decision Engine
    # --------------------------------------------------

    class DummyDecision:

        status = "SUCCESS"
        action = "ALLOW"
        transaction_allowed = True
        authentication_required = False
        voice_required = False
        otp_required = False
        manual_review = False
        message = "Transaction Approved"
        reason = "High Trust"
        audit_log = {}

    class DummyDecisionEngine:

        def decide(self, **kwargs):
            return DummyDecision()

    monkeypatch.setattr(
        server,
        "get_decision_engine",
        lambda: DummyDecisionEngine(),
    )

    # --------------------------------------------------
    # Payload
    # --------------------------------------------------

    payload = {

        "user_id": "USR_1001",

        "transcript": "Transfer five thousand rupees to Rahul",

        "vehicle_speed": 35,

        "engine_running": True,

        "identity": {
            "account_age_days": 120,
            "kyc_verified": True,
            "phone_verified": True,
            "email_verified": True,
            "voice_enrolled": True,
        },

        "biometric": {
            "speaker_similarity": 0.95,
            "liveness_score": 0.98,
            "audio_quality": 0.96,
        },

        "behavior": {
            "behavior_similarity": 0.90,
            "speech_rate_similarity": 0.92,
            "pronunciation_similarity": 0.91,
            "command_familiarity": 0.95,
            "stress_score": 0.10,
        },

        "vehicle": {
            "driver_present": True,
            "seatbelt_fastened": True,
            "engine_running": True,
            "vehicle_speed": 35,
            "location_familiarity": 0.95,
            "time_familiarity": 0.90,
        },

        "history": {
            "failed_attempts": 0,
            "previous_trust_score": 0.94,
        },

        "transaction": {
            "amount": 5000,
            "category": 0,
            "beneficiary_type": 0,
            "purpose": 0,
            "intent_confidence": 0.97,
        },
    }

    response = client.post(
        "/authenticate",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "SUCCESS"
    assert body["action"] == "ALLOW"
    assert body["transaction_allowed"] is True


# ==========================================================
# Invalid Request
# ==========================================================

def test_invalid_request():

    response = client.post(
        "/authenticate",
        json={},
    )

    assert response.status_code in (422, 500)