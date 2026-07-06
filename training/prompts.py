"""
prompts.py
==========
Builds every prompt the pipeline sends to the LLM: system prompt,
batch prompt, single-row prompt, and repair prompt.

This file NEVER calls an API, parses JSON responses, or modifies the
dataset. It only turns (rows, column specs, batch size) into text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from schema import DatasetSchema


@dataclass
class PromptBundle:
    system_prompt: str
    user_prompt: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt(schema: DatasetSchema) -> str:
    label_cols = ", ".join(schema.label_columns) or "(none detected)"
    decision_col = schema.decision_column or "(none detected)"
    score_cols = ", ".join(schema.score_columns) or "(none detected)"

    return (
        "You are a meticulous data-labeling verifier. You are given rows from a "
        "dataset and must check whether the existing labels/scores/decisions are "
        "correct given the available evidence in each row.\n\n"
        f"Label-like columns in this dataset: {label_cols}\n"
        f"Decision column: {decision_col}\n"
        f"Score columns: {score_cols}\n\n"
        "For every row you are given, decide whether the existing values are "
        "correct. If a value is wrong, propose a correction with a short reason "
        "and a confidence between 0 and 1.\n\n"
        "You must respond with ONLY a JSON array (no prose, no markdown fences). "
        "Each element must have this exact shape:\n"
        "{\n"
        '  "row_id": <the row identifier exactly as given>,\n'
        '  "corrections": [\n'
        "    {\n"
        '      "field": <column name>,\n'
        '      "old_value": <original value>,\n'
        '      "new_value": <corrected value>,\n'
        '      "reason": <short reason>,\n'
        '      "confidence": <float 0-1>\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If a row needs no corrections, still include it with an empty "
        '"corrections" list. Preserve row order. Do not add or drop rows. '
        "Do not include any text outside the JSON array."
    )


# ---------------------------------------------------------------------------
# Batch prompt
# ---------------------------------------------------------------------------
def _row_to_dict(row: "pd.Series", id_column: Optional[str], row_index) -> dict:
    d = row.to_dict()
    if id_column and id_column in d:
        d["_row_id"] = d[id_column]
    else:
        d["_row_id"] = row_index
    return d


def build_batch_prompt(rows: "pd.DataFrame", schema: DatasetSchema) -> str:
    """Build the user-turn prompt for a batch of rows."""
    records = [
        _row_to_dict(row, schema.id_column, idx)
        for idx, row in rows.iterrows()
    ]
    import json
    payload = json.dumps(records, indent=2, default=str)
    return (
        f"Here are {len(records)} rows to verify. Each row has a `_row_id` field "
        "you must use as `row_id` in your response.\n\n"
        f"ROWS:\n{payload}\n\n"
        "Respond with the JSON array described in the system prompt, one entry "
        "per row, in the same order."
    )


# ---------------------------------------------------------------------------
# Single-row prompt (used for repairs isolated to one row, or small batches)
# ---------------------------------------------------------------------------
def build_single_row_prompt(row: "pd.Series", schema: DatasetSchema, row_index) -> str:
    record = _row_to_dict(row, schema.id_column, row_index)
    import json
    payload = json.dumps(record, indent=2, default=str)
    return (
        "Here is a single row to verify. Use `_row_id` as `row_id` in your "
        f"response.\n\nROW:\n{payload}\n\n"
        "Respond with a JSON array containing exactly one object, as described "
        "in the system prompt."
    )


# ---------------------------------------------------------------------------
# Repair prompt (sent when a prior response failed validation)
# ---------------------------------------------------------------------------
def build_repair_prompt(original_user_prompt: str, bad_response: str,
                         validation_errors: List[str]) -> str:
    errors_text = "\n".join(f"- {e}" for e in validation_errors)
    return (
        "Your previous response could not be validated and must be corrected.\n\n"
        f"ORIGINAL REQUEST:\n{original_user_prompt}\n\n"
        f"YOUR PREVIOUS RESPONSE:\n{bad_response}\n\n"
        f"VALIDATION ERRORS:\n{errors_text}\n\n"
        "Respond again with ONLY a corrected JSON array that fixes every "
        "validation error above. Follow the exact schema from the system "
        "prompt. Do not include any text outside the JSON array."
    )


# ---------------------------------------------------------------------------
# Convenience builders returning a PromptBundle
# ---------------------------------------------------------------------------
def build_batch_prompt_bundle(rows: "pd.DataFrame", schema: DatasetSchema) -> PromptBundle:
    return PromptBundle(
        system_prompt=build_system_prompt(schema),
        user_prompt=build_batch_prompt(rows, schema),
    )


def build_repair_prompt_bundle(schema: DatasetSchema, original_user_prompt: str,
                                bad_response: str, validation_errors: List[str]) -> PromptBundle:
    return PromptBundle(
        system_prompt=build_system_prompt(schema),
        user_prompt=build_repair_prompt(original_user_prompt, bad_response, validation_errors),
    )
