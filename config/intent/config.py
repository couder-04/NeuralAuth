"""
config.py
=========

Loads deployment-specific constants for the Intent Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("intent_engine.config")

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

_DEFAULTS = {
    # ======================================================
    # LLM Backend
    # ======================================================
    "backend": "LIGHT",
    "light_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "heavy_model": "google/gemma-4-e4b-it",

    # ======================================================
    # Prompt
    # ======================================================
    "prompt_version": "v3.0.0",
    "schema_version": "1.0",

    # ======================================================
    # Generation
    # ======================================================
    "max_retries": 1,
    "max_new_tokens": 128,
    "batch_size": 8,

    # ======================================================
    # Validation
    # ======================================================
    "max_transaction_amount": 10_000_000,
    "max_beneficiary_len": 128,

    "supported_currencies": [
        "INR",
        "USD",
        "EUR",
        "GBP",
    ],

    "valid_intents": [
        "MONEY_TRANSFER",
        "BALANCE_INQUIRY",
        "TRANSACTION_HISTORY",
        "BILL_PAYMENT",
        "UNKNOWN",
    ],

    "valid_categories": [
        "P2P_TRANSFER",
        "MERCHANT_PAYMENT",
        "BILL_PAYMENT",
        "SELF_TRANSFER",
        "UNKNOWN",
    ],

    "valid_purposes": [
        "PERSONAL_TRANSFER",
        "RENT",
        "UTILITY",
        "LOAN_REPAYMENT",
        "SHOPPING",
        "UNKNOWN",
    ],

    "intents_requiring_beneficiary": [
        "MONEY_TRANSFER",
        "BILL_PAYMENT",
    ],
}


@dataclass(frozen=True)
class EngineConfig:
    # ======================================================
    # Backend
    # ======================================================

    backend: str
    light_model: str
    heavy_model: str

    # ======================================================
    # Prompt
    # ======================================================

    prompt_version: str
    schema_version: str

    # ======================================================
    # Generation
    # ======================================================

    max_retries: int
    max_new_tokens: int
    batch_size: int

    # ======================================================
    # Validation
    # ======================================================

    max_transaction_amount: float
    max_beneficiary_len: int

    supported_currencies: List[str] = field(default_factory=list)
    valid_intents: List[str] = field(default_factory=list)
    valid_categories: List[str] = field(default_factory=list)
    valid_purposes: List[str] = field(default_factory=list)
    intents_requiring_beneficiary: List[str] = field(default_factory=list)

    # ======================================================
    # Automatically choose model
    # ======================================================

    @property
    def model_name(self) -> str:

        backend = self.backend.upper()

        if backend == "LIGHT":
            return self.light_model

        if backend == "HEAVY":
            return self.heavy_model

        raise ValueError(
            f"Unknown backend '{self.backend}'. "
            "Expected LIGHT or HEAVY."
        )

    # ======================================================
    # Convenience Sets
    # ======================================================

    @property
    def supported_currencies_set(self):
        return set(self.supported_currencies)

    @property
    def valid_intents_set(self):
        return set(self.valid_intents)

    @property
    def valid_categories_set(self):
        return set(self.valid_categories)

    @property
    def valid_purposes_set(self):
        return set(self.valid_purposes)

    @property
    def intents_requiring_beneficiary_set(self):
        return set(self.intents_requiring_beneficiary)


def load_config(path: Optional[Path] = None) -> EngineConfig:
    """
    Load EngineConfig from config.yaml.
    Falls back to defaults if unavailable.
    """

    path = path or DEFAULT_CONFIG_PATH

    data = dict(_DEFAULTS)

    try:
        import yaml
    except ImportError:
        logger.warning(
            "PyYAML not installed. Using defaults."
        )
        return EngineConfig(**data)

    if not path.exists():
        logger.warning(
            f"{path} not found. Using defaults."
        )
        return EngineConfig(**data)

    try:
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}

        for key in data:
            if key in loaded and loaded[key] is not None:
                data[key] = loaded[key]

    except Exception as exc:
        logger.error(
            f"Failed to load config: {exc}"
        )
        return EngineConfig(**data)

    return EngineConfig(**data)