"""
identity.py
===========

Identity feature generator.

Generated Features
------------------
- account_age_days
- kyc_verified
- phone_verified
- email_verified
- voice_enrolled

These are long-term user profile features and remain
mostly constant across transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class IdentityFeatures:

    account_age_days: int

    kyc_verified: int

    phone_verified: int

    email_verified: int

    voice_enrolled: int

    def to_dict(self) -> Dict:

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class IdentityGenerator:

    def __init__(self, rng: np.random.Generator | None = None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(self) -> IdentityFeatures:

        # --------------------------------------------
        # Account Age
        # Most users have accounts between
        # 6 months and 8 years.
        # --------------------------------------------

        account_age_days = int(

            np.clip(

                self.rng.lognormal(

                    mean=np.log(900),

                    sigma=0.9,

                ),

                30,

                7300,

            )

        )

        # --------------------------------------------
        # KYC Verification
        # Older accounts are much more likely
        # to be KYC verified.
        # --------------------------------------------

        if account_age_days > 365:

            kyc_verified = int(

                self.rng.random() < 0.995

            )

        else:

            kyc_verified = int(

                self.rng.random() < 0.90

            )

        # --------------------------------------------
        # Phone Verification
        # --------------------------------------------

        if kyc_verified:

            phone_verified = int(

                self.rng.random() < 0.995

            )

        else:

            phone_verified = int(

                self.rng.random() < 0.70

            )

        # --------------------------------------------
        # Email Verification
        # --------------------------------------------

        if kyc_verified:

            email_verified = int(

                self.rng.random() < 0.97

            )

        else:

            email_verified = int(

                self.rng.random() < 0.60

            )

        # --------------------------------------------
        # Voice Enrollment
        # Requires high user trust and
        # completed onboarding.
        # --------------------------------------------

        if (

            kyc_verified

            and phone_verified

            and email_verified

        ):

            voice_enrolled = int(

                self.rng.random() < 0.82

            )

        else:

            voice_enrolled = int(

                self.rng.random() < 0.12

            )

        return IdentityFeatures(

            account_age_days=account_age_days,

            kyc_verified=kyc_verified,

            phone_verified=phone_verified,

            email_verified=email_verified,

            voice_enrolled=voice_enrolled,

        )


# ==========================================================
# Public API
# ==========================================================

_generator = IdentityGenerator()


def generate_identity() -> Dict:

    return _generator.generate().to_dict()


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    for _ in range(10):

        print(generate_identity())