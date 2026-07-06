"""
inference/predictor.py

Production inference wrapper for the Authentication Network.

Pipeline
--------
Raw JSON
    ↓
Feature Extractor
    ↓
Feature Tensor
    ↓
Authentication Network
    ↓
Prediction

This module is the only component the API should use.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from engines.authentication_network import (
    AuthenticationNetwork,
    AuthenticationResult,
    ModelConfig,
    Prediction,
)
from engines.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = PROJECT_ROOT / "training" / "models"

CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pth"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
FEATURE_INFO_PATH = ARTIFACT_DIR / "feature_columns.json"
MODEL_INFO_PATH = ARTIFACT_DIR / "model_info.json"
CLASS_MAPPING_PATH = ARTIFACT_DIR / "class_mapping.json"
ENCODER_PATH = ARTIFACT_DIR / "encoder.pkl"


# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------

class AuthenticationPredictor:
    """
    Loads the trained Authentication Network once and serves
    predictions for incoming requests.
    """
    
    model: AuthenticationNetwork
    scaler: StandardScaler
    encoder: Optional[OrdinalEncoder]

    def __init__(self):
        # 1. Device Selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        required_files = [
            CHECKPOINT_PATH,
            SCALER_PATH,
            FEATURE_INFO_PATH,
            MODEL_INFO_PATH,
            CLASS_MAPPING_PATH,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact: {path}")

        # 2. Load Metadata First
        with open(MODEL_INFO_PATH) as f:
            self.model_info = json.load(f)

        with open(FEATURE_INFO_PATH) as f:
            feature_info = json.load(f)
            
        with open(CLASS_MAPPING_PATH) as f:
            self.class_mapping = json.load(f)

        self.feature_order = feature_info["features"]
        self.categorical_cols = feature_info["categorical_cols"]

        # Precompute indices for performance
        self.feature_index = {
            f: i for i, f in enumerate(self.feature_order)
        }
        self.cat_idx = [
            self.feature_index[c] for c in self.categorical_cols
        ]

        # 3. Validate Metadata
        self.expected_features = self.model_info.get("num_features", 0)
        
        # Strict Feature Order Validation
        saved_features = self.model_info.get("feature_order", [])
        if saved_features and saved_features != self.feature_order:
            raise RuntimeError(
                "Feature order mismatch between model_info and feature_info."
            )
            
        if self.expected_features != len(self.feature_order):
            raise RuntimeError(
                f"Feature count mismatch. "
                f"Model expects {self.expected_features} "
                f"but feature file contains {len(self.feature_order)}."
            )
            
        # Strict Categorical Columns Validation
        saved_cats = set(self.model_info.get("categorical_cols", []))
        current_cats = set(self.categorical_cols)
        
        if saved_cats != current_cats:
            raise RuntimeError(
                "Categorical columns mismatch between model_info and feature_info."
            )

        # 4. Load & Validate Encoder
        self.encoder = None
        if len(self.categorical_cols) > 0:
            if not ENCODER_PATH.exists():
                raise RuntimeError("Categorical features exist but encoder.pkl is missing.")
            self.encoder = joblib.load(ENCODER_PATH)
            
            if hasattr(self.encoder, "categories_"):
                if len(self.encoder.categories_) != len(self.categorical_cols):
                    raise RuntimeError("Encoder categories mismatch.")

        # 5. Load & Validate Scaler
        self.scaler = joblib.load(SCALER_PATH)
        if hasattr(self.scaler, "n_features_in_"):
            if self.scaler.n_features_in_ != self.expected_features:
                raise RuntimeError("Scaler feature count mismatch.")

        # 6. Load & Validate Checkpoint
        checkpoint = torch.load(
            CHECKPOINT_PATH, 
            map_location=self.device,
            weights_only=False
        )
        
        required_keys = [
            "model_state_dict", 
            "config", 
            "epoch", 
            "best_macro_f1"
        ]
        
        for key in required_keys:
            if key not in checkpoint:
                raise RuntimeError(f"Checkpoint missing '{key}'")
                
        # Validate Checkpoint Version
        if "checkpoint_version" in checkpoint and "checkpoint_version" in self.model_info:
            if checkpoint["checkpoint_version"] != self.model_info["checkpoint_version"]:
                raise RuntimeError("Checkpoint version mismatch.")

        # 7. Construct and Load Model Manually
        config_data = checkpoint["config"]
        config = ModelConfig(**config_data) if isinstance(config_data, dict) else config_data
        
        if config.num_features != self.expected_features:
            raise RuntimeError("Checkpoint config does not match model_info.")
        
        self.model = AuthenticationNetwork(config)
        
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as e:
            raise RuntimeError("Checkpoint architecture does not match model.") from e
            
        self.model.to(self.device)
        self.model.eval()

        # 8. Dummy Forward & Architecture Warm-up
        dummy = torch.zeros(
            (1, self.expected_features),
            dtype=torch.float32,
            device=self.device,
        )

        try:
            with torch.no_grad():
                output = self.model(dummy)
                # Verify output dimensions against config
                if output.decision_probability.shape[-1] != config.num_decision_classes:
                    raise RuntimeError("Architecture output mismatch detected during warm-up.")
        except Exception as e:
            raise RuntimeError("Model failed warm-up.") from e
            
        logger.info(
            "Loaded Authentication Network (device=%s, checkpoint=%s)",
            self.device,
            CHECKPOINT_PATH.name,
        )

    # --------------------------------------------------------

    def preprocess(self, request: Dict) -> torch.Tensor:
        feature_vector = FeatureExtractor.extract(request)
        feature_dict = FeatureExtractor.to_dict(feature_vector)

        # Safe extraction mapping
        try:
            values = [feature_dict[f] for f in self.feature_order]
        except KeyError as e:
            raise RuntimeError(
                f"Missing feature '{e.args[0]}' from FeatureExtractor."
            )
            
        values = np.array(values, dtype=object).reshape(1, -1)

        # Encode categorical columns exactly as training
        if self.encoder is not None and len(self.cat_idx) > 0:
            values = values.copy()
            cat_values = values[:, self.cat_idx]
            cat_encoded = self.encoder.transform(cat_values)

            values = values.astype(float)
            values[:, self.cat_idx] = cat_encoded

        # Apply StandardScaler
        values = self.scaler.transform(values)

        # Efficient tensor creation avoiding extra copy
        tensor = torch.from_numpy(values.astype(np.float32)).to(self.device)
        
        if tensor.shape[1] != self.expected_features:
            raise RuntimeError("Incorrect feature vector size.")

        return tensor

    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self, request: Dict) -> Prediction:
        """
        Runs deterministic inference. Returns the raw, batched, tensor-based
        `Prediction` -- useful for training/analysis code, but NOT the
        contract downstream engines (Risk/Policy/Decision) should depend
        on. Use `predict_result()` for that.
        """
        features = self.preprocess(request)
        return self.model(features)

    # --------------------------------------------------------

    @torch.no_grad()
    def predict_result(self, request: Dict) -> AuthenticationResult:
        """
        Runs inference and converts the output into the single,
        plain-Python `AuthenticationResult` contract that the Risk Engine,
        Policy Engine, and Decision Engine are designed against (see
        architecture review, Problem 3 -- "Predictor currently combines
        preprocessing, model inference, and result conversion"). Those
        responsibilities are kept as three separate steps here:

            1. preprocess()               raw JSON -> feature tensor
            2. self.model(...)            feature tensor -> Prediction
            3. Prediction.to_result(...)  Prediction -> AuthenticationResult
        """
        start = time.perf_counter()
        features = self.preprocess(request)
        prediction = self.model(features)
        latency_ms = (time.perf_counter() - start) * 1000

        result = prediction.to_result(
            index=0,
            model_version=self.model_info.get("model_version", "unknown"),
        )
        result.latency_ms = round(latency_ms, 3)
        return result

    # --------------------------------------------------------

    @torch.no_grad()
    def predict_dict(self, request: Dict) -> Dict[str, Any]:
        """
        Returns prediction as a Python dictionary. Kept for
        debugging/inspection; API code should use `predict_result()`.
        """
        pred = self.predict(request)
        decision = int(pred.decision.item())
        
        return {
            "trust_score": float(pred.trust_score.item()),
            "risk_score": float(pred.risk_score.item()),
            "confidence": float(pred.confidence.item()),
            "decision": decision,
            "decision_label": self.class_mapping.get(str(decision), "UNKNOWN"),
            "decision_probabilities": pred.decision_probability.squeeze(0).cpu().tolist(),
            "embedding": pred.embedding.squeeze(0).cpu().tolist(),
            "feature_attention": (
                pred.feature_attention.squeeze(0).cpu().tolist()
                if pred.feature_attention is not None
                else None
            ),
        }


# ------------------------------------------------------------
# Lazy Singleton Predictor
# ------------------------------------------------------------

_predictor: Optional[AuthenticationPredictor] = None


def get_predictor() -> AuthenticationPredictor:
    """
    Lazily create the predictor on first use.
    """
    global _predictor

    if _predictor is None:
        _predictor = AuthenticationPredictor()

    return _predictor


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def preprocess(request: Dict) -> torch.Tensor:
    """
    Convert raw request into a feature tensor.
    """
    return get_predictor().preprocess(request)


def predict(request: Dict) -> Dict[str, Any]:
    """
    Run inference and return prediction as a dictionary.
    """
    return get_predictor().predict_dict(request)


def predict_result(request: Dict):
    """
    Run inference and return the unified `AuthenticationResult` contract
    consumed by the Risk / Policy / Decision engines.
    """
    return get_predictor().predict_result(request)