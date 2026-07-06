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

Execution flow (see architecture review, Problem 2 -- the API used to
bypass part of the intended engine flow and import a non-existent
`engines.decision_engine` module):

    Feature Extraction
        -> Authentication Network   (unified AuthenticationResult)
        -> Intent Engine
        -> Risk Engine
        -> Policy Engine
        -> Decision Engine
        -> API Response
"""

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from models.request import TransactionRequest
from models.response import TransactionResponse

from engines.feature_extractor import FeatureExtractor
from inference.predictor import get_predictor
from engines.intent_engine import IntentEngine
from engines.risk_engine import RiskEngine
from engines.policy_engine import PolicyEngine, PolicyInput
from engines.policy_context import (
    classify_location_familiarity,
    classify_time_familiarity,
    parse_request_timestamp,
)
from engines.decision import DecisionEngine

logger = logging.getLogger(__name__)


# ============================================================
# Authentication (API key)
#
# Off by default -- preserves the existing, unauthenticated contract
# for local development and for every existing test, none of which set
# this environment variable. Set `TRANSACTION_ENGINE_API_KEY` before
# exposing this service beyond a fully trusted, isolated environment:
# `/authenticate` runs the full ML pipeline and returns an audit trail
# (see engines/decision/audit.py) for whatever transaction it's given,
# so it should not be reachable by untrusted callers.
#
# Deliberately a thin, dependency-injectable check (`Depends(...)`)
# rather than global middleware, so individual routes opt in and unit
# tests can override the dependency instead of mutating environment
# state.
# ============================================================

API_KEY_ENV_VAR = "TRANSACTION_ENGINE_API_KEY"


def require_api_key(x_api_key: str = Header(default=None)) -> None:
    expected = os.environ.get(API_KEY_ENV_VAR)

    if not expected:
        # No key configured -- authentication is intentionally disabled
        # (dev/test mode). Warned about loudly at startup instead of
        # per-request to avoid log spam.
        return

    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ============================================================
# Engine Loaders
#
# Each loader lazily creates (and memoizes) its engine the first time
# it's called. On its own this is what caused first-request latency --
# the *first* caller of get_predictor()/get_intent_engine()/etc. paid
# the full model-loading cost. `lifespan` below forces every loader to
# run once at process startup, so by the time the server accepts
# traffic all engines are already warm. Tests can still monkeypatch
# these module-level functions directly (see tests/test_api.py) to
# inject fakes without touching real models -- dependency injection is
# unaffected by *when* the real loaders happen to run.
#
# Thread-safety: `lifespan` warms every engine up-front, so in normal
# operation these getters just read an already-populated global and
# never contend on a lock. But the plain
#
#     if engine is None:
#         engine = Engine()
#
# check is still a race if it's ever reached concurrently (e.g. two
# requests both arriving before startup finishes, or the getter being
# called directly in a multi-threaded test/embedding scenario): two
# threads can both see `None`, and both construct + assign an engine,
# leaking one instance and -- for the predictor -- momentarily doubling
# GPU/CPU memory for two loaded models. Each getter below uses the
# standard double-checked-locking pattern with its own dedicated
# `threading.Lock` to guarantee at most one instance is ever
# constructed, while keeping the fast (no-lock) path for the common
# case where the engine is already warm. Each lock only ever guards its
# own single, non-reentrant construction call (engines don't call other
# `get_*_engine()` getters while initializing), so there is no lock
# ordering between them and therefore no deadlock risk.
# ============================================================

_intent_engine_lock = threading.Lock()
_risk_engine_lock = threading.Lock()
_policy_engine_lock = threading.Lock()
_decision_engine_lock = threading.Lock()


def get_intent_engine():
    global intent_engine

    if intent_engine is None:
        with _intent_engine_lock:
            if intent_engine is None:
                intent_engine = IntentEngine()

    return intent_engine


def get_risk_engine():
    global risk_engine

    if risk_engine is None:
        with _risk_engine_lock:
            if risk_engine is None:
                risk_engine = RiskEngine()

    return risk_engine


def get_policy_engine():
    global policy_engine

    if policy_engine is None:
        with _policy_engine_lock:
            if policy_engine is None:
                policy_engine = PolicyEngine()

    return policy_engine


def get_decision_engine():
    global decision_engine

    if decision_engine is None:
        with _decision_engine_lock:
            if decision_engine is None:
                decision_engine = DecisionEngine()

    return decision_engine


intent_engine = None

risk_engine = None

policy_engine = None

decision_engine = None


# ============================================================
# Startup / Shutdown (lifespan)
# ============================================================

# Names (not references!) of every loader that should be warmed up
# before the server accepts traffic. Resolved via `globals()` inside
# `lifespan` at call time -- rather than capturing the function objects
# here -- so that `monkeypatch.setattr(server, "get_predictor", fake)`
# (used both by tests/test_api.py and the startup tests below) is
# honored the same way it already is inside `authenticate()`.
STARTUP_LOADER_NAMES = (
    "get_predictor",
    "get_intent_engine",
    "get_risk_engine",
    "get_policy_engine",
    "get_decision_engine",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained Authentication Network and every engine once, at
    process startup, instead of on the first inbound request.

    If *any* engine fails to initialize (missing model artifacts,
    corrupt checkpoint, architecture mismatch, ...) the exception
    propagates out of this context manager. FastAPI/Starlette turn that
    into a hard startup failure: `uvicorn` refuses to start serving and
    exits non-zero, and `with TestClient(app):` raises the same
    exception -- there is no way to end up with a "running" server that
    is silently missing a model.
    """
    logger.info("Startup: loading models and engines...")

    if not os.environ.get(API_KEY_ENV_VAR):
        logger.warning(
            "%s is not set -- /authenticate is UNAUTHENTICATED. This is "
            "acceptable for local development only; set %s before exposing "
            "this service beyond a trusted, isolated environment.",
            API_KEY_ENV_VAR,
            API_KEY_ENV_VAR,
        )

    for name in STARTUP_LOADER_NAMES:
        loader = globals()[name]
        try:
            loader()
        except Exception:
            logger.critical(
                "Startup failed while running %s() -- server will not start.",
                name,
                exc_info=True,
            )
            raise

    logger.info("Startup complete: all engines are warm.")

    yield

    logger.info("Shutdown complete.")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Transaction Authentication Engine",
    version="1.0.0",
    lifespan=lifespan,
)


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
    dependencies=[Depends(require_api_key)],
)
def authenticate(request: TransactionRequest):

    request_id = uuid.uuid4().hex

    try:

        payload = request.model_dump()

        # ----------------------------------------------------
        # Feature Extraction
        # ----------------------------------------------------

        features = FeatureExtractor.extract(payload)

        predictor = get_predictor()

        # ----------------------------------------------------
        # Authentication Network
        #
        # `predict_result()` returns the single, plain-Python
        # `AuthenticationResult` contract (trust_score, risk_score,
        # confidence, recommended_action, decision_probabilities,
        # attributions, confidence_std) that Risk / Policy / Decision
        # are all designed against -- no more per-caller tensor
        # unpacking or heuristic re-derivation.
        # ----------------------------------------------------

        auth_result = predictor.predict_result(payload)

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

        risk_result = get_risk_engine().evaluate(
            authentication=auth_result,
            intent=intent_prediction,
            features=features,
        )

        # ----------------------------------------------------
        # Policy Engine
        #
        # Field-by-field trace of every `PolicyInput` value (see
        # engines/policy_engine.py for the dataclass and
        # rules/policy_rules.yaml for what each field gates):
        #
        #   trust_score, risk_score, confidence, network_decision
        #       -- from the Authentication Network (`auth_result`).
        #   intent, intent_confidence
        #       -- from the Intent Engine (`intent_prediction`).
        #   risk_level
        #       -- from the Risk Engine (`risk_result`).
        #   transaction_amount, beneficiary_type
        #       -- from the Intent Engine's parse of the transcript.
        #       NOTE: this is the amount/beneficiary the caller *said*,
        #       not necessarily `payload["transaction"]`; if the two
        #       ever need to be reconciled/cross-checked, that belongs
        #       in the Risk Engine's breakdown, not here.
        #   location_familiarity, time_familiarity
        #       -- previously hardcoded to "FAMILIAR"/"NORMAL", which
        #       made the `UnfamiliarLocation`/`OddHourActivity` policy
        #       rules permanently unreachable. Both are now derived
        #       from the *same* `features` FeatureVector already
        #       extracted above (`features.location_familiarity`,
        #       `features.time_familiarity` -- populated from
        #       `request.vehicle`, see engines/feature_extractor.py)
        #       via engines/policy_context.py, which documents the
        #       thresholding assumptions in one place.
        #   previous_trust_score, failed_attempts
        #       -- previously hardcoded to `None`/`0`, which made the
        #       `ManualReviewRepeatedFailures` rule permanently
        #       unreachable. Both come straight from the same
        #       FeatureVector (`request.history`).
        # ----------------------------------------------------

        request_timestamp = parse_request_timestamp(request.timestamp)

        policy_input = PolicyInput(
            trust_score=auth_result.trust_score,
            risk_score=auth_result.risk_score,
            confidence=auth_result.confidence,

            network_decision=auth_result.recommended_action,

            intent=intent_prediction.intent,
            intent_confidence=intent_prediction.confidence,

            risk_level=risk_result.risk_level,

            transaction_amount=intent_prediction.amount,
            beneficiary_type=intent_prediction.beneficiary_type,

            location_familiarity=classify_location_familiarity(
                features.location_familiarity
            ),
            time_familiarity=classify_time_familiarity(
                features.time_familiarity,
                timestamp=request_timestamp,
            ),
            previous_trust_score=features.previous_trust_score,
            failed_attempts=features.failed_attempts,
        )

        policy_result = get_policy_engine().evaluate(policy_input)

        # ----------------------------------------------------
        # Decision Engine
        # ----------------------------------------------------

        decision = get_decision_engine().decide(
            authentication=auth_result,
            risk=risk_result,
            policy=policy_result,
            intent=intent_prediction,
            transaction=payload,
            # Full 31-field FeatureVector (see engines/feature_extractor.py
            # / models/feature_vector.py), included so audit_log carries
            # every engineered feature -- not just the Risk Engine's
            # aggregated breakdown or the model's top-N attributions.
            # Previously computed above and passed to Risk/Policy, but
            # never threaded through here, so it never appeared anywhere
            # in the API response.
            features=features,
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return TransactionResponse(
            status=decision.status,
            action=decision.action.value,
            transaction_allowed=decision.transaction_allowed,
            authentication_required=decision.authentication_required,
            voice_required=decision.voice_required,
            otp_required=decision.otp_required,
            manual_review=decision.manual_review,
            message=decision.message,
            reason=decision.reason,
            audit_log=decision.audit_log,
        )

    except HTTPException:
        # Preserve intentional HTTP errors (e.g. auth failures raised by
        # a dependency) instead of masking them as a generic 500 below.
        raise

    except Exception:
        # Log the full exception server-side (stack trace, internal
        # messages -- e.g. "Feature order mismatch...", file paths,
        # third-party library errors) for debugging/observability, but
        # never return it verbatim to the caller: `str(e)` here could
        # leak internal implementation details (CWE-209). Clients get a
        # generic message plus an opaque request_id they can hand to
        # support/ops to correlate with server-side logs.
        logger.exception(
            "Unhandled error while processing /authenticate request_id=%s",
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing request (request_id={request_id}).",
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
