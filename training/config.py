"""
config.py
=========

Configuration for the Dataset Verification Pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# ---------------------------------------------------------------------------
# Provider Defaults
# ---------------------------------------------------------------------------
#
# NOTE: this must have one entry for every provider `llm_client.py`'s
# `_dispatch()` actually knows how to call (openrouter / claude / gemini).
# `verify_labels.py`'s `--provider` CLI choices are restricted to exactly
# these keys -- if you add a new provider here, add its dispatch method in
# `llm_client.py` first, then add it to the CLI choices.

PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "chat_path": "/chat/completions",
        "default_model": "google/gemini-3.5-flash",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "claude": {
        "base_url": "https://api.anthropic.com",
        "chat_path": "/v1/messages",
        "default_model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "chat_path": "",  # _call_gemini builds its own path from `model`
        "default_model": "gemini-1.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    provider: str = "openrouter"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens: int = 4096
    request_timeout: float = 180.0

    # ------------------------------------------------------------------
    # Retry / Rate Limits
    # ------------------------------------------------------------------

    max_retries: int = 5
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 60.0

    requests_per_minute: int = 40
    parallel_workers: int = 2

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    batch_size: int = 40
    shuffle: bool = False
    max_batches: Optional[int] = None
    resume_from_batch: Optional[int] = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    input_csv: str = "data/dataset.csv"

    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/llm_logs"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    score_min: float = 0.0
    score_max: float = 1.0

    allowed_decisions: Optional[list] = None

    # Explicit schema overrides. `schema.py`'s keyword-based inference is a
    # best-effort fallback for unfamiliar datasets; for a known dataset
    # (like this project's own dataset.csv, where feature names like
    # `transaction_risk` / `previous_trust_score` / `fraud_history` share
    # vocabulary with the *actual* target labels `trust_score` / `risk_score`
    # / `decision` / `confidence`), heuristics alone cannot reliably tell
    # features from labels. Set these explicitly to bypass the heuristic
    # for that field and to give `validator.py` a hard allow-list of columns
    # the LLM is permitted to correct.
    id_column: Optional[str] = None
    decision_column: Optional[str] = None
    label_columns: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Run Control
    # ------------------------------------------------------------------

    resume_enabled: bool = True
    checkpoint_frequency: int = 1

    cost_limit_usd: float = 20.0

    random_seed: int = 42

    dry_run: bool = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self):

        provider_cfg = PROVIDER_DEFAULTS[self.provider]

        if self.model is None:

            self.model = os.getenv(

                "OPENROUTER_MODEL",

                provider_cfg["default_model"],

            )

        if self.base_url is None:

            self.base_url = provider_cfg["base_url"]

        if self.api_key is None:

            self.api_key = os.getenv(

                provider_cfg["api_key_env"]

            )
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: Optional[str] = None, **overrides):

        data = {}

        if config_path:
            path = Path(config_path)

            if not path.exists():
                raise FileNotFoundError(path)

            data = json.loads(path.read_text())

        data.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**data)

    # ------------------------------------------------------------------

    def to_dict(self):
        return asdict(self)

    # ------------------------------------------------------------------

    def save(self, path: str):

        Path(path).write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            )
        )