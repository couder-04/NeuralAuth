"""
merger.py
=========
Takes the original dataframe plus LLM-produced corrections and produces
the final verified dataset, along with a separate corrections-only CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from schema import DatasetSchema
from domain_spec import present_labels


@dataclass
class MergeResult:
    verified_df: "pd.DataFrame"
    corrections_df: "pd.DataFrame"
    rows_corrected: int
    fields_corrected: int
    fields_attempted: int = 0  # includes corrections that failed to apply (row not found / bad field)


def _build_id_lookup(df: "pd.DataFrame", id_column: Optional[str]) -> Optional[Tuple[dict, dict]]:
    """Build an O(1) row_id -> dataframe-index lookup once, instead of
    scanning the whole dataframe (`df[id_column] == row_id`) for every
    correction. Returns (exact_lookup, str_coerced_lookup) or None if there
    is no usable id column."""
    if not id_column or id_column not in df.columns:
        return None
    exact: dict = {}
    coerced: dict = {}
    for idx, val in zip(df.index, df[id_column]):
        exact.setdefault(val, idx)
        coerced.setdefault(str(val), idx)
    return exact, coerced


def _resolve_index(df: "pd.DataFrame", id_column: Optional[str], row_id: Any,
                    id_lookup: Optional[Tuple[dict, dict]] = None):
    """Find the DataFrame index matching a given row_id (id column or raw index)."""
    if id_column and id_column in df.columns:
        if id_lookup is not None:
            exact, coerced = id_lookup
            if row_id in exact:
                return exact[row_id]
            return coerced.get(str(row_id))
        # Fallback (no precomputed lookup supplied): O(n) scan, same
        # behavior as before -- kept for any external caller that doesn't
        # build a lookup table itself.
        matches = df.index[df[id_column] == row_id]
        if len(matches) == 0:
            # try type-coerced match (LLM may return "3" for int id 3)
            matches = df.index[df[id_column].astype(str) == str(row_id)]
        return matches[0] if len(matches) else None
    else:
        try:
            if row_id in df.index:
                return row_id
        except TypeError:
            pass
        return None


def apply_corrections(df: "pd.DataFrame", schema: DatasetSchema,
                       llm_results: List[Dict[str, Any]]) -> MergeResult:
    """Apply a list of {row_id, corrections:[...]} results onto df.

    Returns a new verified DataFrame (df is not mutated) plus a
    corrections-only DataFrame with columns:
        row_id, field, old_value, new_value, reason, confidence

    Defense in depth: only columns in the effective label allow-list
    (domain_spec.present_labels(...) or schema.label_columns) may be
    modified, even if a correction somehow reached this point with a
    feature-column `field` (validator.py already rejects those -- this is
    a second, independent guard so merger.py never mutates a feature
    column like `previous_trust_score`/`transaction_risk`/`fraud_history`
    regardless of how it got here).
    """
    verified_df = df.copy()
    correction_records: List[Dict[str, Any]] = []
    rows_with_corrections = set()

    effective_labels = present_labels(schema.column_names()) or schema.label_columns or []
    id_lookup = _build_id_lookup(verified_df, schema.id_column)

    for result in llm_results:
        row_id = result.get("row_id")
        corrections = result.get("corrections") or []
        if not corrections:
            continue

        idx = _resolve_index(verified_df, schema.id_column, row_id, id_lookup=id_lookup)
        if idx is None:
            # Row couldn't be matched back to the dataframe; skip but record
            # for visibility rather than silently dropping.
            for corr in corrections:
                correction_records.append({
                    "row_id": row_id,
                    "field": corr.get("field"),
                    "old_value": corr.get("old_value"),
                    "new_value": corr.get("new_value"),
                    "reason": corr.get("reason"),
                    "confidence": corr.get("confidence"),
                    "applied": False,
                    "apply_error": "row_id not found in dataset",
                })
            continue

        for corr in corrections:
            field_name = corr.get("field")
            new_value = corr.get("new_value")
            old_value = corr.get("old_value")
            applied = False
            apply_error = None

            if field_name not in verified_df.columns:
                apply_error = f"field '{field_name}' does not exist in dataset"
            elif effective_labels and field_name not in effective_labels:
                apply_error = (
                    f"field '{field_name}' is not an allowed label column "
                    f"(allowed: {effective_labels}); feature columns are read-only "
                    "and were not modified"
                )
            else:
                verified_df.at[idx, field_name] = new_value
                applied = True
                rows_with_corrections.add(idx)

            correction_records.append({
                "row_id": row_id,
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "reason": corr.get("reason"),
                "confidence": corr.get("confidence"),
                "applied": applied,
                "apply_error": apply_error,
            })

    corrections_df = pd.DataFrame(correction_records, columns=[
        "row_id", "field", "old_value", "new_value", "reason", "confidence",
        "applied", "apply_error",
    ])

    fields_applied = sum(1 for r in correction_records if r["applied"])

    return MergeResult(
        verified_df=verified_df,
        corrections_df=corrections_df,
        rows_corrected=len(rows_with_corrections),
        fields_corrected=fields_applied,
        fields_attempted=len(correction_records),
    )


def write_outputs(merge_result: MergeResult, output_dir: str,
                   verified_filename: str = "verified_dataset.csv",
                   corrections_filename: str = "corrections.csv") -> Dict[str, str]:
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    verified_path = out / verified_filename
    corrections_path = out / corrections_filename

    merge_result.verified_df.to_csv(verified_path, index=False)
    merge_result.corrections_df.to_csv(corrections_path, index=False)

    return {"verified_dataset": str(verified_path), "corrections": str(corrections_path)}
