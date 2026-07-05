"""
training/generate_users.py
==========================

Generates synthetic users for the Authentication Network.

Output:
    training/data/users.csv

Each row contains exactly 24 user-level features.
"""

from pathlib import Path

import pandas as pd

from generators.identity import generate_identity
from generators.voice import generate_voice
from generators.behavior import generate_behavior
from generators.vehicle import generate_vehicle
from generators.history import generate_history


# ==========================================================
# Configuration
# ==========================================================

NUM_USERS = 100_000

OUTPUT = Path("training/data/users.csv")


# ==========================================================
# User Generator
# ==========================================================

def generate_user(user_id: int) -> dict:
    """
    Generate one synthetic user.
    """

    row = {

        "user_id": user_id,

        # Identity (5)
        **generate_identity(),

        # Voice (4)
        **generate_voice(),

        # Behaviour (5)
        **generate_behavior(),

        # Vehicle (6)
        **generate_vehicle(),

        # History (4)
        **generate_history(),

    }

    return row


# ==========================================================
# Dataset Generation
# ==========================================================

def main():

    users = [
        generate_user(i + 1)
        for i in range(NUM_USERS)
    ]

    df = pd.DataFrame(users)

    expected_columns = [

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
    print(f"Users Generated : {len(df):,}")
    print(f"Feature Columns : {len(df.columns) - 1}")
    print(f"Saved To        : {OUTPUT}")
    print("=" * 60)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()