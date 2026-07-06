"""
tests/test_api.py

Integration tests for the FastAPI server.

These monkeypatch each engine at the seam the (fixed) pipeline actually
uses: predictor.predict_result() -> AuthenticationResult, Intent Engine
-> Transaction, Risk Engine -> RiskResult, Policy Engine -> PolicyResult,
Decision Engine -> DecisionResult (see api/server.py and the architecture
review's End-to-End Pipeline section).
"""

from fastapi.testclient import TestClient

from api.server import app
from engines.decision.types import DecisionAction, Severity


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

        # Consumed directly by api/server.py when building PolicyInput
        # (see engines/policy_context.py) -- not just via to_dict().
        location_familiarity = 0.95
        time_familiarity = 0.90
        previous_trust_score = 0.94
        failed_attempts = 0

        def to_dict(self):
            return {}

    monkeypatch.setattr(
        server.FeatureExtractor,
        "extract",
        lambda data: DummyFeatureVector(),
    )

    # --------------------------------------------------
    # Fake Predictor -- predict_result() is the seam api/server.py
    # actually calls.
    # --------------------------------------------------

    class DummyAuthResult:

        trust_score = 0.95
        risk_score = 0.10
        confidence = 0.98
        confidence_std = None
        recommended_action = "ALLOW"
        decision_probabilities = {
            "ALLOW": 0.95,
            "VOICE_CHALLENGE": 0.03,
            "VOICE_AND_OTP": 0.01,
            "REJECT": 0.01,
        }
        attributions = {}

    class DummyPredictor:

        def predict_result(self, request):
            return DummyAuthResult()

    monkeypatch.setattr(
        server,
        "get_predictor",
        lambda: DummyPredictor(),
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

        overall_risk = 0.10
        risk_score = 0.10
        risk_level = "LOW"
        confidence = 0.98
        breakdown = {}
        recommended_action = "ALLOW"

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
        priority = "LOW"
        matched_policy = "DefaultAllow"
        matched_rules = []
        rule_trace = []
        policy_score = 0.0
        reason = "No policy rule matched."

    class DummyPolicyEngine:

        def evaluate(self, policy_input):
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
        action = DecisionAction.ALLOW
        severity = Severity.INFO
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
# PolicyInput is derived from real request data, not
# hardcoded placeholders (see engines/policy_context.py)
# ==========================================================

def _authenticate_with_real_policy_engine(monkeypatch, payload):
    """Exercise the real FeatureExtractor + real PolicyEngine, while
    mocking only the Authentication Network / Intent Engine (which need
    trained models) and the Decision Engine (replaced with a spy so the
    test can inspect the exact `PolicyResult` the Policy Engine
    produced). Returns (response, captured_policy_result).
    """
    from api import server

    class DummyAuthResult:
        trust_score = 0.95
        risk_score = 0.10
        confidence = 0.98
        confidence_std = None
        recommended_action = "ALLOW"
        decision_probabilities = {
            "ALLOW": 0.95,
            "VOICE_CHALLENGE": 0.03,
            "VOICE_AND_OTP": 0.01,
            "REJECT": 0.01,
        }
        attributions = {}

    class DummyPredictor:
        def predict_result(self, request):
            return DummyAuthResult()

    monkeypatch.setattr(server, "get_predictor", lambda: DummyPredictor())

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

    monkeypatch.setattr(server, "get_intent_engine", lambda: DummyIntentEngine())

    class DummyRisk:
        overall_risk = 0.10
        risk_score = 0.10
        risk_level = "LOW"
        confidence = 0.98
        breakdown = {}
        recommended_action = "ALLOW"

    class DummyRiskEngine:
        def evaluate(self, **kwargs):
            return DummyRisk()

    monkeypatch.setattr(server, "get_risk_engine", lambda: DummyRiskEngine())

    # NOTE: get_policy_engine / FeatureExtractor.extract are intentionally
    # left untouched -- this is the whole point of the test.

    captured = {}

    class DummyDecision:
        status = "SUCCESS"
        action = DecisionAction.ALLOW
        severity = Severity.INFO
        transaction_allowed = True
        authentication_required = False
        voice_required = False
        otp_required = False
        manual_review = False
        message = "ok"
        reason = "ok"
        audit_log = {}

    class SpyDecisionEngine:
        def decide(self, **kwargs):
            captured["policy"] = kwargs["policy"]
            return DummyDecision()

    monkeypatch.setattr(server, "get_decision_engine", lambda: SpyDecisionEngine())

    response = client.post("/authenticate", json=payload)
    return response, captured.get("policy")


def _base_payload(**overrides):
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
    for section, values in overrides.items():
        payload[section] = {**payload.get(section, {}), **values}
    return payload


def test_unfamiliar_location_rule_is_reachable_from_real_request(monkeypatch):
    """Regression test: `location_familiarity` used to be hardcoded to
    "FAMILIAR" in api/server.py, which made the `UnfamiliarLocation`
    policy rule permanently unreachable no matter what the request
    contained."""
    payload = _base_payload(vehicle={"location_familiarity": 0.1})

    response, policy = _authenticate_with_real_policy_engine(monkeypatch, payload)

    assert response.status_code == 200
    assert policy is not None
    assert policy.matched_policy == "UnfamiliarLocation"
    assert policy.required_action.value == "OTP"


def test_odd_hour_rule_is_reachable_from_real_request(monkeypatch):
    """Regression test: `time_familiarity` used to be hardcoded to
    "NORMAL", which made the `OddHourActivity` policy rule permanently
    unreachable."""
    payload = _base_payload(vehicle={"time_familiarity": 0.1})

    response, policy = _authenticate_with_real_policy_engine(monkeypatch, payload)

    assert response.status_code == 200
    assert policy is not None
    assert policy.matched_policy == "OddHourActivity"
    assert policy.required_action.value == "VOICE_CHALLENGE"


def test_repeated_failures_rule_is_reachable_from_real_request(monkeypatch):
    """Regression test: `failed_attempts` used to be hardcoded to `0`,
    which made the `ManualReviewRepeatedFailures` policy rule
    permanently unreachable."""
    payload = _base_payload(history={"failed_attempts": 5})

    response, policy = _authenticate_with_real_policy_engine(monkeypatch, payload)

    assert response.status_code == 200
    assert policy is not None
    assert policy.matched_policy == "ManualReviewRepeatedFailures"
    assert policy.required_action.value == "MANUAL_REVIEW"


def test_familiar_and_recent_request_reaches_default_allow(monkeypatch):
    """Sanity check: the "happy path" (familiar location, normal hour,
    no failed attempts) still resolves to no extra friction."""
    payload = _base_payload()

    response, policy = _authenticate_with_real_policy_engine(monkeypatch, payload)

    assert response.status_code == 200
    assert policy is not None
    assert policy.required_action.value == "ALLOW"


# ==========================================================
# Invalid Request
# ==========================================================

def test_invalid_request():

    response = client.post(
        "/authenticate",
        json={},
    )

    assert response.status_code in (422, 500)
