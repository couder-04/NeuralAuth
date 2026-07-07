"""
domain_spec.py
==============
Pure domain knowledge for the voice-auth transaction dataset: which
columns are labels vs. features, their valid ranges, which features push
each label up/down, the decision table, and the confidence rubric.

No prompt-building logic lives here -- prompts.py assembles this into
text. Kept separate so the domain spec can be reused, unit-tested, or
swapped for a different dataset without touching prompt-assembly code.

Format notes (token-efficiency):
  - Indicator strength uses ++/+/-/-- instead of prose ("tracks up with")
    so it reads as unambiguous magnitude rather than vague direction.
  - Feature descriptions are single short phrases, not sentences.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Labels -- the ONLY columns `generate_labels.py` writes as output, and the
# ONLY columns the LLM may propose corrections for.
# ---------------------------------------------------------------------------
LABEL_COLUMN_NAMES: List[str] = ["trust_score", "risk_score", "decision", "confidence"]

HARD_CONSTRAINTS: Dict[str, str] = {
    "trust_score": "[0.0, 1.0]",
    "risk_score": "[0.0, 1.0]",
    "confidence": "[0.5, 1.0]  (never below 0.5 on this dataset)",
    "decision": "{0, 1, 2, 3} only  (0=ALLOW, 1=VOICE_CHALLENGE, 2=VOICE_AND_OTP, 3=REJECT)",
}

# indicator strength per label: ++ strong positive, + weak positive,
# - weak negative, -- strong negative. "Positive" = pushes the label UP.
TRUST_SCORE_INDICATORS = """\
  ++ previous_trust_score, speaker_similarity, liveness_score
  +  successful_transactions, account_age_days, kyc_verified, phone_verified, email_verified, voice_enrolled
  -  stress_score, hesitation_score, failed_attempts
  -- spoof_probability"""

RISK_SCORE_INDICATORS = """\
  ++ spoof_probability, fraud_history
  +  failed_attempts, transaction_risk, stress_score, hesitation_score, low liveness_score, large transaction_amount + low account_age_days
  -- high beneficiary_frequency + low transaction_amount + high previous_trust_score (repeated-safe-beneficiary pattern)
Note: trust_score and risk_score are computed independently -- both can be high, both low, or one high one low. They are not required to be opposites."""

DECISION_TABLE = """\
  trust_score   risk_score   decision
  high          low          0 ALLOW
  high          high         2 VOICE_AND_OTP
  low           low          1 VOICE_CHALLENGE
  low           high         3 REJECT
Overrides (apply regardless of the table above):
  - spoof_probability >= ~0.85 or liveness_score <= ~0.10  -> decision must not be 0 (ALLOW)
  - fraud_history clearly present + failed_attempts high    -> decision must be >= 2
"high"/"low" mean relative to the row's other signals, not a fixed universal cutoff -- use judgment, not a hardcoded number."""

CONFIDENCE_RUBRIC = """\
  0.90 - 1.00  evidence strongly and unanimously supports the decision (all features agree)
  0.75 - 0.90  evidence supports the decision but with 1 mildly conflicting feature
  0.50 - 0.75  evidence is genuinely mixed/conflicting, or trust_score and risk_score are both mid/high at once"""

LABEL_MEANINGS: Dict[str, str] = {
    "trust_score": "Output of the Trust Engine. Indicators:\n" + TRUST_SCORE_INDICATORS,
    "risk_score": "Output of the Risk Engine. Indicators:\n" + RISK_SCORE_INDICATORS,
    "decision": "Categorical outcome, driven by trust_score + risk_score together. Decision table:\n" + DECISION_TABLE,
    "confidence": (
        "Model's confidence in its own decision. Rubric:\n" + CONFIDENCE_RUBRIC +
        "\nNot the same column as the `llm_confidence` feature below."
    ),
}

# ---------------------------------------------------------------------------
# Features -- raw input evidence. READ-ONLY. Never a valid correction target.
# Three of these have names that closely resemble label names -- flagged
# explicitly since that's the most likely source of past LLM confusion.
# ---------------------------------------------------------------------------
FEATURE_COLUMN_SPECS: Dict[str, str] = {
    "user_id": "unique row/account id",
    "account_age_days": "1-5000, days since account creation",
    "kyc_verified": "0/1",
    "phone_verified": "0/1",
    "email_verified": "0/1",
    "voice_enrolled": "0/1, has enrolled voiceprint",
    "speaker_similarity": "0-1, live voice vs enrolled voiceprint match",
    "liveness_score": "0-1, confidence audio is live (not replay/deepfake)",
    "audio_quality": "0-1, technical audio clarity",
    "spoof_probability": "0-1, probability audio is spoofed",
    "speech_rate_similarity": "0-1, speech rate vs. baseline",
    "pronunciation_similarity": "0-1, pronunciation vs. baseline",
    "command_familiarity": "0-1, how typical this command is for the user",
    "stress_score": "0-1, vocal stress",
    "hesitation_score": "0-1, hesitation/pause pattern strength",
    "vehicle_speed": "0-250 km/h",
    "engine_running": "0/1",
    "location_familiarity": "0-1, how familiar the current location is",
    "time_familiarity": "0-1, how typical the time-of-day is",
    "driver_present": "0/1",
    "seatbelt_fastened": "0/1",
    "previous_trust_score": "0-1, RAW INPUT to the trust engine -- NOT the trust_score label",
    "failed_attempts": "0-50, recent failed auth attempts",
    "successful_transactions": "0-10000, lifetime successful transactions",
    "fraud_history": "numeric, past-fraud signal/count for this account",
    "transaction_amount": "0-10,000,000, amount requested",
    "transaction_category": "categorical string",
    "beneficiary_type": "categorical string",
    "beneficiary_frequency": "0-1, how often this beneficiary has been paid before",
    "transaction_risk": "0-1, RAW INPUT to the risk engine -- NOT the risk_score label",
    "intent_type": "categorical string, detected transaction intent",
    "llm_confidence": "0-1, RAW confidence from an upstream intent classifier -- NOT the confidence label",
}

NAME_COLLISION_WARNING = (
    "Name-collision trap: previous_trust_score, transaction_risk, and llm_confidence are "
    "FEATURES. They are not trust_score, risk_score, or confidence. Never propose a "
    "correction for the three feature names even though they resemble label names."
)


def present_labels(available_columns: List[str]) -> List[str]:
    return [c for c in LABEL_COLUMN_NAMES if c in available_columns]