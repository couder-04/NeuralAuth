"""
checkpoint.py
=============
Allows an interrupted verification run to resume where it left off.

Stores: last completed batch, rows completed, timestamp, API usage,
current outputs (accumulated corrections), and a checksum of the
input file (so we notice if the dataset changed underneath us).

Public API: save(), load(), clear(), resume()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import checksum_of_file


CHECKPOINT_FILENAME = "checkpoint.json"


@dataclass
class CheckpointState:
    last_completed_batch: int = -1          # -1 = nothing completed yet
    rows_completed: int = 0
    timestamp: str = ""
    input_checksum: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_api_calls: int = 0
    corrections: List[Dict[str, Any]] = field(default_factory=list)  # accumulated
    verified_row_ids: List[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        return cls(**d)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, input_csv_path: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.checkpoint_dir / CHECKPOINT_FILENAME
        self.input_csv_path = input_csv_path

    # ------------------------------------------------------------------
    def save(self, state: CheckpointState) -> None:
        state.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self.input_csv_path:
            state.input_checksum = checksum_of_file(self.input_csv_path)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state.to_dict(), indent=2, default=str))
        tmp_path.replace(self.path)  # atomic-ish write

    def load(self) -> Optional[CheckpointState]:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text())
        return CheckpointState.from_dict(data)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def resume(self, expect_input_checksum: Optional[str] = None) -> CheckpointState:
        """Load an existing checkpoint, or return a fresh one if none/invalid.

        If the input file's checksum has changed since the checkpoint was
        written, the checkpoint is considered stale and a fresh state is
        returned instead (so we don't silently apply corrections computed
        against a different dataset).
        """
        state = self.load()
        if state is None:
            return CheckpointState()

        if expect_input_checksum and state.input_checksum:
            if state.input_checksum != expect_input_checksum:
                return CheckpointState()  # stale checkpoint, start fresh

        return state
