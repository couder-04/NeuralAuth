"""
training/generate_users.py
==========================

Generates synthetic users for the Authentication Network.

Output:
    training/data/users.csv           -- exactly 24 user-level features
                                          (unchanged schema, no fraud
                                          columns -- a model should never
                                          be handed the answer key).
    training/data/fraud_context.csv   -- side-channel ground truth: which
                                          fraud scenario (if any) each
                                          user_id was generated under, and
                                          how strongly it impacted each
                                          subsystem. Consumed by
                                          `generate_transactions.py` (to
                                          keep the same user's fraud
                                          context consistent when
                                          generating their transaction)
                                          and by `analyze_fraud.py` (for
                                          dataset validation/reporting).

Each user is assigned a `FraudContext` (see `generators/fraud.py`) before
any subsystem is generated, and that context is threaded into every
subsystem generator that plausibly correlates with it (voice, behavior,
vehicle, history). Genuine users (~94-95% by default) get a zero-impact
context, which is mathematically identical to the old "fraudulent=False"
path.
"""

import argparse
from pathlib import Path

import pandas as pd

from generators.identity import generate_identity
from generators.voice import generate_voice
from generators.behavior import generate_behavior
from generators.vehicle import generate_vehicle
from generators.history import generate_history
from generators.fraud import (
    FRAUD_CONTEXT_COLUMNS,
    FraudContext,
    FraudGeneratorConfig,
    FraudScenarioSampler,
)


# ==========================================================
# Configuration
# ==========================================================

NUM_USERS = 100_000

OUTPUT = Path("training/data/users.csv")
FRAUD_OUTPUT = Path("training/data/fraud_context.csv")


# ==========================================================
# User Generator
# ==========================================================

def generate_user(user_id: int, fraud_context: FraudContext) -> dict:
    """
    Generate one synthetic user under the given fraud context.
    """

    row = {

        "user_id": user_id,

        # Identity (5) -- deliberately NOT fraud-coupled: KYC/verification
        # state reflects the account's real onboarding history, which an
        # attacker taking over or replaying against that account does not
        # change.
        **generate_identity(),

        # Voice (4)
        **generate_voice(fraud_context=fraud_context),

        # Behaviour (5)
        **generate_behavior(fraud_context=fraud_context),

        # Vehicle (6)
        **generate_vehicle(fraud_context=fraud_context),

        # History (4)
        **generate_history(fraud_context=fraud_context),

    }

    return row


# ==========================================================
# Dataset Generation
# ==========================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-users",
        type=int,
        default=NUM_USERS,
        help=f"Number of synthetic users to generate (default: {NUM_USERS:,}).",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=None,
        help="Override FraudGeneratorConfig.fraud_rate (default: 0.055).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the fraud sampler's random seed (default: 42).",
    )
    args = parser.parse_args()

    fraud_config = FraudGeneratorConfig()
    if args.fraud_rate is not None:
        fraud_config.fraud_rate = args.fraud_rate
    if args.seed is not None:
        fraud_config.random_seed = args.seed

    sampler = FraudScenarioSampler(fraud_config)

    users = []
    fraud_rows = []

    for i in range(args.num_users):
        user_id = i + 1
        ctx = sampler.sample()
        users.append(generate_user(user_id, ctx))
        fraud_rows.append(ctx.to_row(user_id))

    df = pd.DataFrame(users)
    fraud_df = pd.DataFrame(fraud_rows, columns=list(FRAUD_CONTEXT_COLUMNS))

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

    fraud_df.to_csv(
        FRAUD_OUTPUT,
        index=False,
    )

    fraud_counts = fraud_df["scenario"].value_counts(normalize=True)

    print("=" * 60)
    print(f"Users Generated      : {len(df):,}")
    print(f"Feature Columns      : {len(df.columns) - 1}")
    print(f"Saved To             : {OUTPUT}")
    print(f"Fraud Context Saved  : {FRAUD_OUTPUT}")
    print("-" * 60)
    print("Fraud Scenario Distribution:")
    for scenario, pct in fraud_counts.items():
        print(f"  - {scenario:<32} {pct:.2%}")
    print("=" * 60)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()
