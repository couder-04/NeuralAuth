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
from typing import Dict, Optional

import numpy as np

from generators.fraud import FraudContext, blend


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
        fraud_context: Optional[FraudContext] = None,
    ) -> HistoryFeatures:

        ctx = FraudContext.resolve(fraudulent, fraud_context)

        # --------------------------------------------
        # Fraud History
        # --------------------------------------------

        impact_fh = ctx.feature_impact("history", "fraud_history")
        p_fraud_history = blend(0.02, 0.65, impact_fh)

        fraud_history = self.rng.choice(
            [0, 1],
            p=[1.0 - p_fraud_history, p_fraud_history],
        )

        # --------------------------------------------
        # Successful Transactions
        # --------------------------------------------

        impact_succ = ctx.feature_impact(
            "history", "successful_transactions"
        )

        successful_transactions = int(

            np.clip(

                self.rng.normal(
                    blend(1800, 120, impact_succ),
                    blend(900, 80, impact_succ),
                ),

                blend(20, 0, impact_succ),
                blend(10000, 600, impact_succ),

            )

        )

        # --------------------------------------------
        # Failed Attempts
        # --------------------------------------------

        impact_fail = ctx.feature_impact("history", "failed_attempts")

        failed_attempts = int(

            np.clip(

                self.rng.poisson(
                    blend(0.6, 4.0, impact_fail)
                ),

                0,
                blend(5, 15, impact_fail),

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
    fraud_context: Optional[FraudContext] = None,
) -> Dict:

    return _generator.generate(
        fraudulent=fraudulent,
        fraud_context=fraud_context,
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