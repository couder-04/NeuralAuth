"""
training/generate_transactions.py
=================================

Generates transaction-level features for every user.

Input:
    training/data/users.csv

Output:
    training/data/transactions.csv

Appends 7 transaction features to every user.
"""

from pathlib import Path

import pandas as pd

from generators.transaction import generate_transaction
from generators.intent import generate_intent


# ==========================================================
# Configuration
# ==========================================================

INPUT = Path("training/data/users.csv")

OUTPUT = Path("training/data/transactions.csv")


# ==========================================================
# Transaction Generator
# ==========================================================

def generate_transaction_row(user: dict) -> dict:
    """
    Generate one transaction for a user.
    """

    transaction = generate_transaction()

    intent = generate_intent(

        transaction["transaction_category"]

    )

    row = {

        **user,

        # Transaction (5)
        **transaction,

        # Intent (2)
        **intent,

    }

    return row


# ==========================================================
# Dataset Generation
# ==========================================================

def main():

    users = pd.read_csv(INPUT)

    # NOTE: `DataFrame.iterrows()` yields each row as a `pd.Series`, which
    # forces every value in that row to a single common dtype (float64,
    # since the frame also has float columns). That silently corrupts every
    # integer column (user_id, kyc_verified, account_age_days, ...) into
    # floats. `to_dict("records")` preserves each column's original scalar
    # dtype per cell, so we use that instead.
    rows = [
        generate_transaction_row(user)
        for user in users.to_dict("records")
    ]

    df = pd.DataFrame(rows)

    expected_columns = [

        # Metadata
        "user_id",

        # Identity
        "account_age_days",
        "kyc_verified",
        "phone_verified",
        "email_verified",
        "voice_enrolled",

        # Voice
        "speaker_similarity",
        "liveness_score",
        "audio_quality",
        "spoof_probability",

        # Behaviour
        "speech_rate_similarity",
        "pronunciation_similarity",
        "command_familiarity",
        "stress_score",
        "hesitation_score",

        # Vehicle
        "vehicle_speed",
        "engine_running",
        "location_familiarity",
        "time_familiarity",
        "driver_present",
        "seatbelt_fastened",

        # History
        "previous_trust_score",
        "failed_attempts",
        "successful_transactions",
        "fraud_history",

        # Transaction (5)
        "transaction_amount",
        "transaction_category",
        "beneficiary_type",
        "beneficiary_frequency",
        "transaction_risk",

        # Intent (2)
        "intent_type",
        "llm_confidence",

    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Column mismatch.\nExpected:\n{expected_columns}\n\nGot:\n{list(df.columns)}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 60)
    print(f"Transactions Generated : {len(df):,}")
    print(f"Feature Columns        : {len(df.columns) - 1}")
    print(f"Saved To              : {OUTPUT}")
    print("=" * 60)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()