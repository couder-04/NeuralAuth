"""
behavior.py
===========

Behavioral biometric feature generator.

Generated Features
------------------
- speech_rate_similarity
- pronunciation_similarity
- command_familiarity
- stress_score
- hesitation_score

Simulates behavioral biometrics produced by a
voice-command authentication system.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class BehaviorFeatures:

    speech_rate_similarity: float

    pronunciation_similarity: float

    command_familiarity: float

    stress_score: float

    hesitation_score: float

    def to_dict(self) -> Dict:

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class BehaviorGenerator:

    def __init__(self, rng: np.random.Generator | None = None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(
        self,
        fraudulent: bool = False,
    ) -> BehaviorFeatures:

        # --------------------------------------------
        # Command Familiarity
        # --------------------------------------------

        if fraudulent:

            command_familiarity = np.clip(

                self.rng.normal(
                    0.35,
                    0.18,
                ),

                0,
                1,

            )

        else:

            command_familiarity = np.clip(

                self.rng.normal(
                    0.88,
                    0.08,
                ),

                0,
                1,

            )

        # --------------------------------------------
        # Stress Score
        # --------------------------------------------

        if fraudulent:

            stress_score = np.clip(

                self.rng.normal(
                    0.72,
                    0.15,
                ),

                0,
                1,

            )

        else:

            stress_score = np.clip(

                self.rng.normal(
                    0.22,
                    0.10,
                ),

                0,
                1,

            )

        # --------------------------------------------
        # Hesitation Score
        # Correlated with stress
        # --------------------------------------------

        hesitation_score = np.clip(

            0.70 * stress_score
            + self.rng.normal(
                0.12,
                0.08,
            ),

            0,
            1,

        )

        # --------------------------------------------
        # Pronunciation Similarity
        # Higher for genuine users
        # Lower if stressed
        # --------------------------------------------

        if fraudulent:

            base_pronunciation = 0.62

        else:

            base_pronunciation = 0.93

        pronunciation_similarity = np.clip(

            base_pronunciation
            - 0.18 * stress_score
            + self.rng.normal(
                0,
                0.03,
            ),

            0,
            1,

        )

        # --------------------------------------------
        # Speech Rate Similarity
        # Depends on hesitation + familiarity
        # --------------------------------------------

        speech_rate_similarity = np.clip(

            0.55 * command_familiarity
            + 0.35 * pronunciation_similarity
            - 0.20 * hesitation_score
            + self.rng.normal(
                0,
                0.03,
            ),

            0,
            1,

        )

        return BehaviorFeatures(

            speech_rate_similarity=round(
                float(speech_rate_similarity),
                4,
            ),

            pronunciation_similarity=round(
                float(pronunciation_similarity),
                4,
            ),

            command_familiarity=round(
                float(command_familiarity),
                4,
            ),

            stress_score=round(
                float(stress_score),
                4,
            ),

            hesitation_score=round(
                float(hesitation_score),
                4,
            ),

        )


# ==========================================================
# Public API
# ==========================================================

_generator = BehaviorGenerator()


def generate_behavior(
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

        print(generate_behavior())

    print("\nFraudulent Users\n")

    for _ in range(5):

        print(generate_behavior(fraudulent=True))