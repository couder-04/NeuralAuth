"""
intent.py
=========

Intent feature generator.

Generated Features
------------------
- intent_type
- llm_confidence

Intent is derived from the transaction category to
simulate an LLM parsing spoken commands.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Intent Encoding
# ==========================================================

BALANCE_INQUIRY = 0
MONEY_TRANSFER = 1
BILL_PAYMENT = 2
TRANSACTION_HISTORY = 3


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class IntentFeatures:

    intent_type: int

    llm_confidence: float

    def to_dict(self):

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class IntentGenerator:

    def __init__(self, rng=None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(
        self,
        transaction_category: int,
        fraudulent: bool = False,
    ) -> IntentFeatures:

        # Intent matches transaction category

        intent_type = transaction_category

        # --------------------------------------------
        # LLM Confidence
        # --------------------------------------------

        if fraudulent:

            confidence = np.clip(

                self.rng.normal(
                    0.82,
                    0.10,
                ),

                0.50,
                1.00,

            )

        else:

            confidence = np.clip(

                self.rng.normal(
                    0.96,
                    0.03,
                ),

                0.70,
                1.00,

            )

        return IntentFeatures(

            intent_type=int(intent_type),

            llm_confidence=round(
                float(confidence),
                4,
            ),

        )


# ==========================================================
# Public API
# ==========================================================

_generator = IntentGenerator()


def generate_intent(
    transaction_category: int,
    fraudulent: bool = False,
) -> Dict:

    return _generator.generate(
        transaction_category=transaction_category,
        fraudulent=fraudulent,
    ).to_dict()


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    for category in range(4):

        print(generate_intent(category))