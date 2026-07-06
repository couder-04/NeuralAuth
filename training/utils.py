"""
utils.py
========
Shared, stateless helper functions used across the pipeline:
logging setup, timers, progress display, hashing, UUIDs,
CSV helpers, token estimation, pretty printing.

No pipeline logic lives here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str = "verify_labels", log_dir: Optional[str] = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a logger that writes to stdout and, if given, a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    """Context manager that logs (or prints) how long a block took."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[timer] {label} took {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


class Stopwatch:
    """Simple reusable stopwatch for cumulative timing (e.g. total API time)."""

    def __init__(self):
        self._total = 0.0
        self._start: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is not None:
            self._total += time.perf_counter() - self._start
            self._start = None

    @property
    def total(self) -> float:
        return self._total


# ---------------------------------------------------------------------------
# Progress display (lightweight, no external deps)
# ---------------------------------------------------------------------------
def print_progress(current: int, total: int, prefix: str = "", width: int = 30) -> None:
    if total <= 0:
        return
    frac = min(current / total, 1.0)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({frac*100:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Hashing / IDs
# ---------------------------------------------------------------------------
def stable_hash(value: str) -> str:
    """Deterministic short hash, useful for checksums / row fingerprints."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def new_uuid() -> str:
    return str(uuid.uuid4())


def checksum_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Token / cost estimation (rough heuristics; providers differ)
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Very rough token estimate: ~4 characters per token (English-ish text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# Rough $ per 1K tokens, (input, output). Update as pricing changes.
COST_PER_1K_TOKENS = {
    "deepseek-chat": (0.00027, 0.0011),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "qwen-plus": (0.0004, 0.0012),
    "claude-sonnet-4-6": (0.003, 0.015),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = COST_PER_1K_TOKENS.get(model, (0.0, 0.0))
    return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------
def safe_json_loads(text: str):
    """Try to parse JSON, stripping common LLM wrapper artifacts (```json fences)."""
    if text is None:
        raise ValueError("Cannot parse JSON from None")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove a leading language tag like "json\n"
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def pretty(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except TypeError:
        return str(obj)


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
