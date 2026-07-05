"""
dataset.py
==========

PyTorch Dataset for the Adaptive Transaction Authentication Network.

Responsibilities
----------------
1. Load raw dataset.
2. Build FeatureVector using FeatureExtractor.
3. Encode categorical features.
4. Normalize numerical features.
5. Convert everything into tensors.

Output
------
(feature_tensor, target)
"""

from typing import List, Dict

import math
import torch
from torch.utils.data import Dataset

from engines.feature_extractor import FeatureExtractor


# ============================================================
# Encoders
# ============================================================

CATEGORY_MAP = {
    "P2P_TRANSFER": 0,
    "MERCHANT_PAYMENT": 1,
    "BILL_PAYMENT": 2,
    "SELF_TRANSFER": 3,
    "UNKNOWN": 4,
}

BENEFICIARY_MAP = {
    "SAVED": 0,
    "NEW": 1,
    "UNKNOWN": 2,
}

PURPOSE_MAP = {
    "PERSONAL_TRANSFER": 0,
    "RENT": 1,
    "UTILITY": 2,
    "LOAN_REPAYMENT": 3,
    "SHOPPING": 4,
    "UNKNOWN": 5,
}


# ============================================================
# Dataset
# ============================================================

class TransactionDataset(Dataset):

    def __init__(self, samples: List[Dict]):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------------

    @staticmethod
    def normalize_percentage(value: float):

        return value / 100.0

    @staticmethod
    def normalize_amount(amount: float):

        return math.log1p(amount)

    @staticmethod
    def normalize_speed(speed: float):

        return speed / 200.0

    @staticmethod
    def normalize_days(days: float):

        return min(days / 3650.0, 1.0)

    # --------------------------------------------------------

    def build_tensor(self, feature):

        x = [

            # =====================================================
            # Identity
            # =====================================================

            self.normalize_days(feature.account_age_days),

            float(feature.kyc_verified),
            float(feature.phone_verified),
            float(feature.email_verified),
            float(feature.voice_enrolled),

            # =====================================================
            # Biometrics
            # =====================================================

            self.normalize_percentage(feature.speaker_similarity),
            self.normalize_percentage(feature.liveness_score),
            self.normalize_percentage(feature.audio_quality),

            # =====================================================
            # Behavior
            # =====================================================

            self.normalize_percentage(feature.behavior_similarity),
            self.normalize_percentage(feature.speech_rate_similarity),
            self.normalize_percentage(feature.pronunciation_similarity),
            self.normalize_percentage(feature.command_familiarity),
            self.normalize_percentage(feature.stress_score),

            # =====================================================
            # Vehicle
            # =====================================================

            float(feature.driver_present),
            float(feature.seatbelt_fastened),
            float(feature.engine_running),

            self.normalize_speed(feature.vehicle_speed),

            self.normalize_percentage(feature.location_familiarity),
            self.normalize_percentage(feature.time_familiarity),

            # =====================================================
            # History
            # =====================================================

            feature.failed_attempts / 10.0,

            self.normalize_percentage(feature.previous_trust_score),

            # =====================================================
            # Intent
            # =====================================================

            self.normalize_percentage(feature.intent_confidence),

            self.normalize_amount(feature.transaction_amount),

            CATEGORY_MAP.get(
                feature.transaction_category,
                CATEGORY_MAP["UNKNOWN"]
            ),

            BENEFICIARY_MAP.get(
                feature.beneficiary_type,
                BENEFICIARY_MAP["UNKNOWN"]
            ),

            PURPOSE_MAP.get(
                feature.transaction_purpose,
                PURPOSE_MAP["UNKNOWN"]
            )

        ]

        return torch.tensor(
            x,
            dtype=torch.float32
        )

    # --------------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        feature = FeatureExtractor.extract(sample)

        x = self.build_tensor(feature)

        y = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return x, y