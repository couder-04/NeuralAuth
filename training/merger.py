"""
merger.py
=========
Takes the original dataframe plus LLM-produced corrections and produces
the final verified dataset, along with a separate corrections-only CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from schema import DatasetSchema


@dataclass
class MergeResult:
    verified_df: "pd.DataFrame"
    corrections_df: "pd.DataFrame"
    rows_corrected: int
    fields_corrected: int


def _resolve_index(df: "pd.DataFrame", id_column: Optional[str], row_id: Any):
    """Find the DataFrame index matching a given row_id (id column or raw index)."""
    if id_column and id_column in df.columns:
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
    """
    verified_df = df.copy()
    correction_records: List[Dict[str, Any]] = []
    rows_with_corrections = set()

    for result in llm_results:
        row_id = result.get("row_id")
        corrections = result.get("corrections") or []
        if not corrections:
            continue

        idx = _resolve_index(verified_df, schema.id_column, row_id)
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

    return MergeResult(
        verified_df=verified_df,
        corrections_df=corrections_df,
        rows_corrected=len(rows_with_corrections),
        fields_corrected=len(correction_records),
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
