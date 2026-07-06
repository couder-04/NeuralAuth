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
from typing import Dict, Optional

import numpy as np

from generators.fraud import FraudContext, blend


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
        fraud_context: Optional[FraudContext] = None,
    ) -> VoiceFeatures:

        ctx = FraudContext.resolve(fraudulent, fraud_context)

        # --------------------------------------------
        # Audio Quality
        # Blended between the genuine and fraud
        # distributions by this scenario's voice
        # impact (0 = fully genuine, 1 = fully fraud).
        # --------------------------------------------

        impact_aq = ctx.feature_impact("voice", "audio_quality")

        audio_quality = np.clip(

            self.rng.normal(
                blend(0.92, 0.72, impact_aq),
                blend(0.06, 0.15, impact_aq),
            ),

            0,
            1,

        )

        # --------------------------------------------
        # Liveness
        # --------------------------------------------

        impact_live = ctx.feature_impact("voice", "liveness_score")

        liveness_score = np.clip(

            self.rng.normal(
                blend(0.97, 0.35, impact_live),
                blend(0.03, 0.18, impact_live),
            ),

            0,
            1,

        )

        # --------------------------------------------
        # Speaker Similarity
        # Depends on audio quality + liveness
        # --------------------------------------------

        impact_sim = ctx.feature_impact("voice", "speaker_similarity")

        base_similarity = blend(0.90, 0.45, impact_sim)

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

    for _ in range(20):

        print(generate_voice())

    print("\nFraudulent Users\n")

    for _ in range(20):

        print(generate_voice(fraudulent=True))