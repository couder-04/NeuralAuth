"""
transaction.py
==============

Transaction feature generator.

Generated Features
------------------
- transaction_amount
- transaction_category
- beneficiary_type
- beneficiary_frequency
- transaction_risk
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Transaction Categories
# ==========================================================

BALANCE_INQUIRY = 0
MONEY_TRANSFER = 1
BILL_PAYMENT = 2
TRANSACTION_HISTORY = 3


# ==========================================================
# Beneficiary Types
# ==========================================================

KNOWN = 0
NEW = 1


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class TransactionFeatures:

    transaction_amount: float

    transaction_category: int

    beneficiary_type: int

    beneficiary_frequency: float

    transaction_risk: float

    def to_dict(self):

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class TransactionGenerator:

    def __init__(self, rng=None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(
        self,
        fraudulent: bool = False,
    ) -> TransactionFeatures:

        # --------------------------------------------
        # Transaction Category
        # --------------------------------------------

        category = self.rng.choice(

            [BALANCE_INQUIRY,
             MONEY_TRANSFER,
             BILL_PAYMENT,
             TRANSACTION_HISTORY],

            p=[0.10, 0.55, 0.20, 0.15]

        )

        # --------------------------------------------
        # Amount
        # --------------------------------------------

        if category == BALANCE_INQUIRY:

            amount = 0.0

        elif category == TRANSACTION_HISTORY:

            amount = 0.0

        elif category == BILL_PAYMENT:

            amount = np.clip(

                self.rng.normal(
                    4500,
                    2500,
                ),

                100,
                25000,

            )

        else:

            if fraudulent:

                amount = np.clip(

                    self.rng.normal(
                        120000,
                        60000,
                    ),

                    500,
                    500000,

                )

            else:

                amount = np.clip(

                    self.rng.normal(
                        25000,
                        18000,
                    ),

                    100,
                    250000,

                )

        # --------------------------------------------
        # Beneficiary Type
        # --------------------------------------------

        if category in (
            BALANCE_INQUIRY,
            TRANSACTION_HISTORY,
        ):

            beneficiary_type = KNOWN

        else:

            if fraudulent:

                beneficiary_type = self.rng.choice(
                    [KNOWN, NEW],
                    p=[0.25, 0.75],
                )

            else:

                beneficiary_type = self.rng.choice(
                    [KNOWN, NEW],
                    p=[0.85, 0.15],
                )

        # --------------------------------------------
        # Beneficiary Frequency
        # --------------------------------------------

        if beneficiary_type == NEW:

            beneficiary_frequency = np.clip(

                self.rng.normal(
                    0.08,
                    0.05,
                ),

                0,
                0.30,

            )

        else:

            beneficiary_frequency = np.clip(

                self.rng.normal(
                    0.82,
                    0.12,
                ),

                0,
                1,

            )

        # --------------------------------------------
        # Transaction Risk
        # --------------------------------------------

        risk = (

            0.10

            + min(amount / 500000, 1.0) * 0.45

            + (beneficiary_type == NEW) * 0.25

            + self.rng.normal(
                0,
                0.05,
            )

        )

        transaction_risk = np.clip(
            risk,
            0,
            1,
        )

        return TransactionFeatures(

            transaction_amount=round(
                float(amount),
                2,
            ),

            transaction_category=int(category),

            beneficiary_type=int(beneficiary_type),

            beneficiary_frequency=round(
                float(beneficiary_frequency),
                4,
            ),

            transaction_risk=round(
                float(transaction_risk),
                4,
            ),

        )


# ==========================================================
# Public API
# ==========================================================

_generator = TransactionGenerator()


def generate_transaction(
    fraudulent: bool = False,
) -> Dict:

    return _generator.generate(
        fraudulent=fraudulent,
    ).to_dict()


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    print("\nLegitimate Transactions\n")

    for _ in range(5):

        print(generate_transaction())

    print("\nFraudulent Transactions\n")

    for _ in range(5):

        print(generate_transaction(fraudulent=True))