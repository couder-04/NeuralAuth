"""
history.py
==========

Historical profile feature generator.

Generated Features
------------------
- previous_trust_score
- failed_attempts
- successful_transactions
- fraud_history
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class HistoryFeatures:

    previous_trust_score: float

    failed_attempts: int

    successful_transactions: int

    fraud_history: int

    def to_dict(self):

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class HistoryGenerator:

    def __init__(self, rng=None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(
        self,
        fraudulent: bool = False,
    ) -> HistoryFeatures:

        # --------------------------------------------
        # Fraud History
        # --------------------------------------------

        if fraudulent:

            fraud_history = self.rng.choice(
                [0, 1],
                p=[0.35, 0.65],
            )

        else:

            fraud_history = self.rng.choice(
                [0, 1],
                p=[0.98, 0.02],
            )

        # --------------------------------------------
        # Successful Transactions
        # --------------------------------------------

        if fraudulent:

            successful_transactions = int(

                np.clip(

                    self.rng.normal(
                        120,
                        80,
                    ),

                    0,
                    600,

                )

            )

        else:

            successful_transactions = int(

                np.clip(

                    self.rng.normal(
                        1800,
                        900,
                    ),

                    20,
                    10000,

                )

            )

        # --------------------------------------------
        # Failed Attempts
        # --------------------------------------------

        if fraudulent:

            failed_attempts = int(

                np.clip(

                    self.rng.poisson(4),

                    0,
                    15,

                )

            )

        else:

            failed_attempts = int(

                np.clip(

                    self.rng.poisson(0.6),

                    0,
                    5,

                )

            )

        # --------------------------------------------
        # Previous Trust Score
        # --------------------------------------------

        trust = (

            0.45
            + 0.00018 * successful_transactions
            - 0.05 * failed_attempts
            - 0.28 * fraud_history
            + self.rng.normal(
                0,
                0.04,
            )

        )

        previous_trust_score = np.clip(
            trust,
            0,
            1,
        )

        return HistoryFeatures(

            previous_trust_score=round(
                float(previous_trust_score),
                4,
            ),

            failed_attempts=failed_attempts,

            successful_transactions=successful_transactions,

            fraud_history=fraud_history,

        )


# ==========================================================
# Public API
# ==========================================================

_generator = HistoryGenerator()


def generate_history(
    fraudulent: bool = False,
) -> Dict:

    return _generator.generate(
        fraudulent=fraudulent,
    ).to_dict()


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    print("\nLegitimate Users\n")

    for _ in range(5):

        print(generate_history())

    print("\nFraudulent Users\n")

    for _ in range(5):

        print(generate_history(fraudulent=True))