"""
schema.py
=========
Automatically infer the dataset schema so the pipeline is reusable
across datasets without hardcoding column names like `trust_score`
or `decision`.

Matching notes (see _matches_any/_is_pure_label_name below):
  - Keyword matching is done on WHOLE TOKENS (word-boundary), not raw
    substrings. This avoids false positives like "id" matching inside
    "confidence" (conf-ID-ence), which previously misclassified the
    `confidence`/`llm_confidence` columns as an id column.
  - For the score/trust/risk keyword group specifically, a column is only
    classified as a label/score column if its ENTIRE normalized name is
    composed of recognized label vocabulary tokens (e.g. "trust_score" ->
    ["trust", "score"], both in-vocab -> label). A column with extra
    qualifier tokens (e.g. "previous_trust_score" -> ["previous", "trust",
    "score"], "previous" not in-vocab) is treated as a feature instead --
    this is what keeps raw inputs like `previous_trust_score`,
    `transaction_risk`, and `fraud_history` from being misclassified as
    labels just because they share a keyword with the real label columns.

Explicit overrides (Config.id_column / decision_column / label_columns) can
always be passed to `infer_schema()` to bypass the heuristic entirely for a
known dataset -- see that function's docstring.

Public API:
    infer_schema(df, id_column=None, decision_column=None, label_columns=None) -> DatasetSchema
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic keyword groups. Column names are matched case-insensitively,
# on whole tokens (or whole token-phrases for multi-word keywords), with
# underscores/hyphens/whitespace all treated as token separators.
# ---------------------------------------------------------------------------
ID_KEYWORDS = ["id", "uuid", "transaction_id", "txn_id", "record_id", "row_id"]
SCORE_KEYWORDS = ["score", "confidence", "probability", "prob"]
TRUST_KEYWORDS = ["trust"]
RISK_KEYWORDS = ["risk", "fraud"]
DECISION_KEYWORDS = ["decision", "label", "verdict", "outcome", "class", "target"]
REASON_KEYWORDS = ["reason", "explanation", "rationale", "notes"]
TEXT_EVIDENCE_KEYWORDS = ["transcript", "text", "description", "comment", "note", "evidence"]

# Vocabulary used for the "pure label name" check on the score/trust/risk
# keyword group only (see module docstring). Flattened to individual tokens.
_LABELISH_VOCAB = set()
for _kw in SCORE_KEYWORDS + TRUST_KEYWORDS + RISK_KEYWORDS:
    _LABELISH_VOCAB.update(_kw.replace("_", " ").replace("-", " ").split())


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


def _tokens(norm_name: str) -> List[str]:
    return norm_name.split()


def _matches_any(tokens: List[str], keywords: List[str]) -> bool:
    """True if any keyword (single word or multi-word phrase) appears in
    `tokens` as a contiguous, whole-token match. Word-boundary matching --
    NOT substring matching -- so e.g. the keyword "id" matches the token
    "id" but never matches inside the token "confidence"."""
    for kw in keywords:
        kw_tokens = _tokens(_normalize(kw))
        n = len(kw_tokens)
        if n == 0:
            continue
        if any(tokens[i:i + n] == kw_tokens for i in range(len(tokens) - n + 1)):
            return True
    return False


def _is_pure_labelish_name(tokens: List[str]) -> bool:
    """True only if EVERY token in the column name is drawn from the
    score/trust/risk label vocabulary (see _LABELISH_VOCAB). This is what
    distinguishes a real label column (`trust_score`, `risk_score`,
    `confidence`) from a feature that merely shares a keyword
    (`previous_trust_score`, `transaction_risk`, `fraud_history`,
    `llm_confidence`, `spoof_probability`, ...): those have extra
    qualifier tokens ("previous", "transaction", "history", "llm",
    "spoof") that are not part of the label vocabulary."""
    return len(tokens) > 0 and all(t in _LABELISH_VOCAB for t in tokens)


def _classify_column(name: str, series: "pd.Series") -> ColumnSpec:
    norm = _normalize(name)
    tokens = _tokens(norm)
    dtype = str(series.dtype)

    if _matches_any(tokens, ID_KEYWORDS):
        role = "id"
    elif _matches_any(tokens, DECISION_KEYWORDS):
        role = "decision"
    elif _matches_any(tokens, REASON_KEYWORDS):
        role = "reason"
    elif (
        _matches_any(tokens, SCORE_KEYWORDS)
        or _matches_any(tokens, TRUST_KEYWORDS)
        or _matches_any(tokens, RISK_KEYWORDS)
    ) and _is_pure_labelish_name(tokens):
        role = "score"
    elif _matches_any(tokens, TEXT_EVIDENCE_KEYWORDS):
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


def infer_schema(
    df: "pd.DataFrame",
    id_column: Optional[str] = None,
    decision_column: Optional[str] = None,
    label_columns: Optional[List[str]] = None,
) -> DatasetSchema:
    """Infer a DatasetSchema from a pandas DataFrame using column-name heuristics.

    Falls back gracefully: if no obvious id column exists, the DataFrame's
    index will be used as the row identifier by callers (batcher/merger).

    Explicit overrides (typically `Config.id_column` / `Config.decision_column`
    / `Config.label_columns`) take precedence over the heuristic whenever
    supplied and present in `df.columns`. This is the intended way to handle
    a known dataset where feature names share vocabulary with label names
    (e.g. `transaction_risk` vs. `risk_score`) and the heuristic alone can't
    reliably tell them apart. If an override is supplied but the named
    column doesn't exist in `df`, it is ignored (with a warning) and the
    heuristic result is used instead.
    """
    columns = [_classify_column(col, df[col]) for col in df.columns]
    available = list(df.columns)

    id_candidates = [c.name for c in columns if c.role == "id"]
    decision_candidates = [c.name for c in columns if c.role == "decision"]
    score_candidates = [c.name for c in columns if c.role == "score"]
    reason_candidates = [c.name for c in columns if c.role == "reason"]
    evidence_candidates = [c.name for c in columns if c.role == "evidence"]
    label_candidates = [c.name for c in columns if c.is_label]

    resolved_id = id_candidates[0] if id_candidates else None
    if id_column is not None:
        if id_column in available:
            resolved_id = id_column
        else:
            logger.warning(
                "infer_schema: id_column override '%s' not found in dataframe "
                "columns; falling back to heuristic result (%s).",
                id_column, resolved_id,
            )

    resolved_decision = decision_candidates[0] if decision_candidates else None
    if decision_column is not None:
        if decision_column in available:
            resolved_decision = decision_column
        else:
            logger.warning(
                "infer_schema: decision_column override '%s' not found in "
                "dataframe columns; falling back to heuristic result (%s).",
                decision_column, resolved_decision,
            )

    resolved_labels = label_candidates
    resolved_scores = score_candidates
    if label_columns is not None:
        missing = [c for c in label_columns if c not in available]
        if missing:
            logger.warning(
                "infer_schema: label_columns override contains columns not "
                "present in the dataframe, ignoring them: %s", missing,
            )
        resolved_labels = [c for c in label_columns if c in available]
        # score_columns (used for numeric range validation) is every
        # overridden label except the decision column, which is validated
        # separately via `allowed_decisions` membership, not a numeric range.
        resolved_scores = [c for c in resolved_labels if c != resolved_decision]

        # Keep each ColumnSpec's role/is_label in sync with the override so
        # that anything reading `schema.columns` directly (not just the
        # convenience lists) sees a consistent picture.
        label_set = set(resolved_labels)
        for c in columns:
            if c.name == resolved_decision:
                c.role = "decision"
                c.is_label = True
            elif c.name in label_set:
                c.role = "score"
                c.is_label = True
            elif c.is_label:
                # Was heuristically a label/score but not in the explicit
                # override list -- demote it back to a plain feature.
                c.role = "other"
                c.is_label = False

    return DatasetSchema(
        columns=columns,
        id_column=resolved_id,
        decision_column=resolved_decision,
        score_columns=resolved_scores,
        reason_columns=reason_candidates,
        evidence_columns=evidence_candidates,
        label_columns=resolved_labels,
    )
