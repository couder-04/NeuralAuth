"""
server.py
=========

REST API for the Transaction Authentication Engine.

Responsibilities
----------------
1. Receive transaction requests.
2. Execute the complete authentication pipeline.
3. Return the final decision.

Contains NO business logic.
"""


from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException

from models.request import TransactionRequest
from models.response import TransactionResponse

from engines.feature_extractor import FeatureExtractor
from inference.predictor import get_predictor
from engines.intent_engine import IntentEngine
from engines.risk_engine import RiskEngine
from engines.policy_engine import PolicyEngine
from engines.decision_engine import DecisionEngine


# ============================================================
# Lazy Engine Loaders
# ============================================================

def get_intent_engine():
    global intent_engine

    if intent_engine is None:
        intent_engine = IntentEngine()

    return intent_engine


def get_risk_engine():
    global risk_engine

    if risk_engine is None:
        risk_engine = RiskEngine()

    return risk_engine


def get_policy_engine():
    global policy_engine

    if policy_engine is None:
        policy_engine = PolicyEngine()

    return policy_engine


def get_decision_engine():
    global decision_engine

    if decision_engine is None:
        decision_engine = DecisionEngine()

    return decision_engine

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Transaction Authentication Engine",
    version="1.0.0",
)



intent_engine = None

risk_engine = None

policy_engine = None

decision_engine = None


# ============================================================
# Health
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Transaction Authentication Engine",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Authentication Endpoint
# ============================================================

@app.post(
    "/authenticate",
    response_model=TransactionResponse,
)
def authenticate(request: TransactionRequest):

    try:

        # ----------------------------------------------------
        # Feature Extraction
        # ----------------------------------------------------

        features = FeatureExtractor.extract(
            request.model_dump()
        )

        predictor = get_predictor()

        # ----------------------------------------------------
        # Authentication Network
        # ----------------------------------------------------

        auth_prediction = predictor.predict(request.model_dump())

        # ----------------------------------------------------
        # Intent Engine
        # ----------------------------------------------------

        parsed_intent = get_intent_engine().parse(
            request.transcript
        )

        intent_prediction = parsed_intent.transaction

        # ----------------------------------------------------
        # Risk Engine
        # ----------------------------------------------------

        risk_prediction = get_risk_engine().evaluate(

            authentication=auth_prediction,

            intent=intent_prediction,

            features=features,

        )

        # ----------------------------------------------------
        # Policy Engine
        # ----------------------------------------------------

        from engines.policy_engine import PolicyEngine, PolicyInput
        from engines.authentication_network import DECISION_LABELS

        policy_input = PolicyInput(
            trust_score=auth_prediction.trust_score.squeeze().item(),
            risk_score=auth_prediction.risk_score.squeeze().item(),
            confidence=auth_prediction.confidence.squeeze().item(),

            network_decision=DECISION_LABELS[
                auth_prediction.decision.squeeze().item()
            ],

            intent=intent_prediction.intent,
            intent_confidence=intent_prediction.confidence,

            risk_level=risk_prediction.risk_level,

            transaction_amount=intent_prediction.amount,
            beneficiary_type=intent_prediction.beneficiary_type,

            # Until you implement these features
            location_familiarity="FAMILIAR",
            time_familiarity="NORMAL",
            previous_trust_score=None,
            failed_attempts=0,
        )

        policy_result = get_policy_engine().evaluate(policy_input)

        # ----------------------------------------------------
        # Decision Engine
        # ----------------------------------------------------

        decision = get_decision_engine().decide(

            authentication=auth_prediction,

            risk=risk_prediction,

            policy=policy_result,

            intent=intent_prediction,

            transaction=request.model_dump(),

        )
        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return TransactionResponse(

            status=decision.status,

            action=decision.action,

            transaction_allowed=decision.transaction_allowed,

            authentication_required=decision.authentication_required,

            voice_required=decision.voice_required,

            otp_required=decision.otp_required,

            manual_review=decision.manual_review,

            message=decision.message,

            reason=decision.reason,

            audit_log=decision.audit_log,

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e),

        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )