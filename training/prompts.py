"""
prompts.py
==========
Builds every prompt the pipeline sends to the LLM: system prompt,
batch prompt, single-row prompt, and repair prompt.

This file NEVER calls an API, parses JSON responses, or modifies the
dataset. It only turns (rows, column specs, batch size) into text.

Layout (split for maintainability -- see each module's docstring):
  domain_spec.py    -- label/feature meanings, hard constraints, decision
                       table, confidence rubric (pure data, no prompt text)
  examples.py       -- worked calibration examples
  output_schema.py  -- the JSON response contract
  prompts.py (here) -- assembles the above into system/user prompts,
                       plus the reasoning process and row-batching logic

Token-efficiency note: the system prompt is IDENTICAL across every batch
in a run (it doesn't depend on row content, only on which columns are
present). If your provider supports prompt caching (e.g. Anthropic's
`cache_control` on the system block), mark this system prompt as
cacheable in llm_client.py -- that turns "N batches x full system prompt"
into "1x full price + N x cached read", which is the actual fix for
token cost at scale. This file keeps the prompt as short as it can be
made without losing the fixes that address the label/feature confusion,
since caching (not further trimming) is the higher-leverage lever here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from schema import DatasetSchema
from domain_spec import (
    LABEL_COLUMN_NAMES,
    HARD_CONSTRAINTS,
    LABEL_MEANINGS,
    FEATURE_COLUMN_SPECS,
    NAME_COLLISION_WARNING,
    present_labels,
)
from examples import WORKED_EXAMPLES
from output_schema import response_format_block


@dataclass
class PromptBundle:
    system_prompt: str
    user_prompt: str


# ---------------------------------------------------------------------------
# Reference block: labels + features actually present in this dataset
# ---------------------------------------------------------------------------
def _reference_block(schema: "DatasetSchema", label_cols: List[str]) -> str:
    available = schema.column_names()
    feature_cols = [c for c in available if c not in label_cols]

    lines = ["LABELS you may correct (hard range in brackets):"]
    for name in label_cols:
        constraint = HARD_CONSTRAINTS.get(name, "(no constraint on file)")
        meaning = LABEL_MEANINGS.get(name, "(no spec available -- treat cautiously)")
        lines.append(f"\n{name} {constraint}\n{meaning}")

    lines.append("\nFEATURES, read-only evidence, never a correction target:")
    for name in feature_cols:
        desc = FEATURE_COLUMN_SPECS.get(name, "raw feature, do not correct")
        lines.append(f"  - {name}: {desc}")

    lines.append(f"\n{NAME_COLLISION_WARNING}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt(schema: DatasetSchema) -> str:
    available = schema.column_names()
    label_cols = present_labels(available) or schema.label_columns
    allowed_fields = ", ".join(label_cols) or "(none detected)"

    reference = _reference_block(schema, label_cols)

    return f"""\
You are a label-verification auditor for a voice-driven in-vehicle transaction
authentication system. Each row has raw evidence (features) and four
model-generated labels. Decide whether the labels are consistent with the
features in that SAME row. Treat every row independently -- never use one
row's values to judge another row, and never infer dataset-wide statistics.

{reference}

REASONING STEPS (do internally, do not output them):
1. From the features alone, what would you expect trust_score to be?
2. From the features alone, what would you expect risk_score to be?
3. Given your expected trust_score/risk_score, what decision follows (see table)?
4. Given how much the features agree or conflict, what confidence follows (see rubric)?
5. Compare each expectation to the row's actual label. Only propose a correction
   for a MATERIAL mismatch -- not because a nearby value would also be reasonable
   (see example E). Justify each correction with 2+ specific feature values (see
   examples A-C).

WORKED EXAMPLES:
{WORKED_EXAMPLES}

{response_format_block(allowed_fields)}"""


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


def _split_row_for_prompt(record: dict, label_cols: List[str]) -> dict:
    """Reshape a row dict into explicit `features` / `labels_to_verify`
    groups so the LLM doesn't have to re-derive the split from column
    names on every row."""
    row_id = record.get("_row_id")
    features = {k: v for k, v in record.items() if k not in label_cols and k != "_row_id"}
    labels = {k: v for k, v in record.items() if k in label_cols}
    return {"_row_id": row_id, "features": features, "labels_to_verify": labels}


def build_batch_prompt(rows: "pd.DataFrame", schema: DatasetSchema) -> str:
    """Build the user-turn prompt for a batch of rows."""
    available = schema.column_names()
    label_cols = present_labels(available) or schema.label_columns

    records = [
        _split_row_for_prompt(
            _row_to_dict(row, schema.id_column, idx), label_cols
        )
        for idx, row in rows.iterrows()
    ]
    import json
    payload = json.dumps(records, indent=2, default=str)
    return (
        f"{len(records)} rows to verify, each independent of the others. Each row is "
        "split into `features` (read-only) and `labels_to_verify` (the only fields you "
        "may correct). Use `_row_id` as `row_id` in your response.\n\n"
        f"ROWS:\n{payload}\n\n"
        "Respond with the JSON array described in the system prompt, one entry per row, "
        "same order."
    )


# ---------------------------------------------------------------------------
# Single-row prompt (used for repairs isolated to one row, or small batches)
# ---------------------------------------------------------------------------
def build_single_row_prompt(row: "pd.Series", schema: DatasetSchema, row_index) -> str:
    available = schema.column_names()
    label_cols = present_labels(available) or schema.label_columns
    record = _split_row_for_prompt(
        _row_to_dict(row, schema.id_column, row_index), label_cols
    )
    import json
    payload = json.dumps(record, indent=2, default=str)
    return (
        "Single row to verify, split into `features` (read-only) and `labels_to_verify` "
        f"(the only fields you may correct). Use `_row_id` as `row_id`.\n\nROW:\n{payload}\n\n"
        "Respond with a JSON array containing exactly one object, per the system prompt."
    )


# ---------------------------------------------------------------------------
# Repair prompt (sent when a prior response failed validation)
# ---------------------------------------------------------------------------
def build_repair_prompt(original_user_prompt: str, bad_response: str,
                         validation_errors: List[str]) -> str:
    errors_text = "\n".join(f"- {e}" for e in validation_errors)
    return (
        "Your previous response failed validation and must be corrected.\n\n"
        f"ORIGINAL REQUEST:\n{original_user_prompt}\n\n"
        f"YOUR PREVIOUS RESPONSE:\n{bad_response}\n\n"
        f"VALIDATION ERRORS:\n{errors_text}\n\n"
        "Common causes: `field` was a feature column, not a label; a value violated its "
        "hard constraint (see ranges in the system prompt); a key was renamed/omitted; "
        "output included non-JSON text. Respond again with ONLY a corrected JSON array "
        "that fixes every error above, following the exact schema from the system prompt."
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