"""
voice.py
========

Voice biometric feature generator.

Generated Features
------------------
- speaker_similarity
- liveness_score
- audio_quality
- spoof_probability

These features simulate the output of a production
Speaker Verification + Anti-Spoofing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class VoiceFeatures:

    speaker_similarity: float

    liveness_score: float

    audio_quality: float

    spoof_probability: float

    def to_dict(self) -> Dict:

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class VoiceGenerator:

    def __init__(self, rng: np.random.Generator | None = None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(
        self,
        fraudulent: bool = False,
    ) -> VoiceFeatures:

        # --------------------------------------------
        # Audio Quality
        # --------------------------------------------

        if fraudulent:

            audio_quality = np.clip(

                self.rng.normal(
                    0.72,
                    0.15,
                ),

                0,
                1,

            )

        else:

            audio_quality = np.clip(

                self.rng.normal(
                    0.92,
                    0.06,
                ),

                0,
                1,

            )

        # --------------------------------------------
        # Liveness
        # --------------------------------------------

        if fraudulent:

            liveness_score = np.clip(

                self.rng.normal(
                    0.35,
                    0.18,
                ),

                0,
                1,

            )

        else:

            liveness_score = np.clip(

                self.rng.normal(
                    0.97,
                    0.03,
                ),

                0,
                1,

            )

        # --------------------------------------------
        # Speaker Similarity
        # Depends on audio quality + liveness
        # --------------------------------------------

        if fraudulent:

            base_similarity = 0.45

        else:

            base_similarity = 0.90

        speaker_similarity = np.clip(

            base_similarity
            + 0.08 * audio_quality
            + 0.07 * liveness_score
            + self.rng.normal(
                0,
                0.03,
            ),

            0,
            1,

        )

        # --------------------------------------------
        # Spoof Probability
        # Inversely related to similarity/liveness
        # --------------------------------------------

        spoof_probability = np.clip(

            1.0
            - (
                0.50 * liveness_score
                + 0.35 * speaker_similarity
                + 0.15 * audio_quality
            )
            + self.rng.normal(
                0,
                0.02,
            ),

            0,
            1,

        )

        return VoiceFeatures(

            speaker_similarity=round(
                float(speaker_similarity),
                4,
            ),

            liveness_score=round(
                float(liveness_score),
                4,
            ),

            audio_quality=round(
                float(audio_quality),
                4,
            ),

            spoof_probability=round(
                float(spoof_probability),
                4,
            ),

        )


# ==========================================================
# Public API
# ==========================================================

_generator = VoiceGenerator()


def generate_voice(
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

    for _ in range(20):

        print(generate_voice())

    print("\nFraudulent Users\n")

    for _ in range(20):

        print(generate_voice(fraudulent=True))