"""
schema.py
=========
Automatically infer the dataset schema so the pipeline is reusable
across datasets without hardcoding column names like `trust_score`
or `decision`.

Public API:
    infer_schema(df) -> DatasetSchema
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Heuristic keyword groups. Column names are matched case-insensitively,
# with underscores/hyphens treated as spaces.
# ---------------------------------------------------------------------------
ID_KEYWORDS = ["id", "uuid", "transaction_id", "txn_id", "record_id", "row_id"]
SCORE_KEYWORDS = ["score", "confidence", "probability", "prob"]
TRUST_KEYWORDS = ["trust"]
RISK_KEYWORDS = ["risk", "fraud"]
DECISION_KEYWORDS = ["decision", "label", "verdict", "outcome", "class", "target"]
REASON_KEYWORDS = ["reason", "explanation", "rationale", "notes"]
TEXT_EVIDENCE_KEYWORDS = ["transcript", "text", "description", "comment", "note", "evidence"]


@dataclass
class ColumnSpec:
    name: str
    role: str                # one of: id, score, decision, reason, evidence, other
    dtype: str = "unknown"    # pandas dtype as string
    is_label: bool = False    # True if this is likely the thing being verified
    sample_values: List = field(default_factory=list)


@dataclass
class DatasetSchema:
    columns: List[ColumnSpec]
    id_column: Optional[str]
    decision_column: Optional[str]
    score_columns: List[str]
    reason_columns: List[str]
    evidence_columns: List[str]
    label_columns: List[str]

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get(self, name: str) -> Optional[ColumnSpec]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


def _normalize(name: str) -> str:
    return re.sub(r"[_\-\s]+", " ", name.strip().lower())


def _matches_any(norm_name: str, keywords: List[str]) -> bool:
    return any(kw in norm_name for kw in keywords)


def _classify_column(name: str, series: "pd.Series") -> ColumnSpec:
    norm = _normalize(name)
    dtype = str(series.dtype)

    if _matches_any(norm, ID_KEYWORDS):
        role = "id"
    elif _matches_any(norm, DECISION_KEYWORDS):
        role = "decision"
    elif _matches_any(norm, REASON_KEYWORDS):
        role = "reason"
    elif _matches_any(norm, SCORE_KEYWORDS) or _matches_any(norm, TRUST_KEYWORDS) or _matches_any(norm, RISK_KEYWORDS):
        role = "score"
    elif _matches_any(norm, TEXT_EVIDENCE_KEYWORDS):
        role = "evidence"
    else:
        role = "other"

    is_label = role in ("decision", "score")

    sample_values = []
    try:
        sample_values = series.dropna().unique().tolist()[:5]
    except TypeError:
        sample_values = []

    return ColumnSpec(name=name, role=role, dtype=dtype, is_label=is_label,
                       sample_values=sample_values)


def infer_schema(df: "pd.DataFrame") -> DatasetSchema:
    """Infer a DatasetSchema from a pandas DataFrame using column-name heuristics.

    Falls back gracefully: if no obvious id column exists, the DataFrame's
    index will be used as the row identifier by callers (batcher/merger).
    """
    columns = [_classify_column(col, df[col]) for col in df.columns]

    id_candidates = [c.name for c in columns if c.role == "id"]
    decision_candidates = [c.name for c in columns if c.role == "decision"]
    score_candidates = [c.name for c in columns if c.role == "score"]
    reason_candidates = [c.name for c in columns if c.role == "reason"]
    evidence_candidates = [c.name for c in columns if c.role == "evidence"]
    label_candidates = [c.name for c in columns if c.is_label]

    return DatasetSchema(
        columns=columns,
        id_column=id_candidates[0] if id_candidates else None,
        decision_column=decision_candidates[0] if decision_candidates else None,
        score_columns=score_candidates,
        reason_columns=reason_candidates,
        evidence_columns=evidence_candidates,
        label_columns=label_candidates,
    )
