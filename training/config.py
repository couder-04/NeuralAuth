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
# This must have exactly one entry for every provider `llm_client.py`'s
# `_dispatch()` knows how to call, which must in turn be exactly the set of
# `verify_labels.py --provider` CLI choices -- the three lists (this dict's
# keys, `_dispatch`'s routing, and the CLI `choices=`) are kept in sync
# deliberately; add/remove a provider in all three places together.
#
# openrouter / openai / deepseek / qwen all speak the same OpenAI-compatible
# chat-completions wire format (`_call_openai_compatible` in llm_client.py
# handles all four); claude and gemini have their own bespoke request/
# response shapes and dedicated `_call_anthropic` / `_call_gemini` methods.
#
# `model_env`: the environment variable consulted for a default model name
# when `Config.model` isn't set explicitly. Provider-specific (previously
# every provider incorrectly fell back to reading `OPENROUTER_MODEL`).

PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "chat_path": "/chat/completions",
        "default_model": "google/gemini-3.5-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
    },
    "claude": {
        "base_url": "https://api.anthropic.com",
        "chat_path": "/v1/messages",
        "default_model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model_env": "CLAUDE_MODEL",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "chat_path": "",  # _call_gemini builds its own path from `model`
        "default_model": "gemini-1.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat_path": "/chat/completions",
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "chat_path": "/chat/completions",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
    },
    "qwen": {
        # DashScope's OpenAI-compatible mode.
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_path": "/chat/completions",
        "default_model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model_env": "QWEN_MODEL",
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
    # Explicit override for which batch index to resume from. Normally
    # resume position comes from the checkpoint (`state.last_completed_batch
    # + 1`); set this to force a specific start batch instead (verify_labels.py
    # takes the max of the two so this can only skip forward, never replay
    # already-checkpointed batches). None = use the checkpoint-derived value.
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

    # Default is filled in __post_init__ from domain_spec.ALLOWED_DECISIONS
    # (the {0,1,2,3} canonical decision set) rather than left as None, so
    # decision-value validation is enforced by default instead of silently
    # disabled. Pass an explicit empty list to opt back out.
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

    # When True, verify_labels.py deletes any existing checkpoint before
    # running. A real dataclass field (previously injected at runtime via
    # `config.__dict__["clear_checkpoint"] = True`, which bypassed
    # `Config.load`'s override mechanism and wasn't visible to
    # `to_dict()`/`save()`).
    clear_checkpoint: bool = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self):

        provider_cfg = PROVIDER_DEFAULTS[self.provider]

        if self.model is None:

            self.model = os.getenv(

                provider_cfg["model_env"],

                provider_cfg["default_model"],

            )

        if self.base_url is None:

            self.base_url = provider_cfg["base_url"]

        if self.api_key is None:

            self.api_key = os.getenv(

                provider_cfg["api_key_env"]

            )

        if self.allowed_decisions is None:
            # Import kept local to __post_init__ to avoid a module-level
            # dependency cycle risk between config.py and domain_spec.py.
            from domain_spec import ALLOWED_DECISIONS
            self.allowed_decisions = list(ALLOWED_DECISIONS)
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