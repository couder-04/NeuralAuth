"""
validator.py
============
Protects the pipeline from LLM mistakes.

Validates:
    - JSON parses at all
    - Overall schema (array of {row_id, corrections})
    - Each correction has the required fields, no extras
    - Decision values are within allowed set (if known)
    - Score/confidence values are within range
    - No missing / duplicate row_ids
    - Response batch size matches the request
    - Row ordering matches the request

On failure, returns a ValidationResult with is_valid=False and a list of
human-readable errors that prompts.build_repair_prompt can use to ask the
LLM to fix its answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from schema import DatasetSchema
from utils import safe_json_loads

REQUIRED_CORRECTION_FIELDS = {"field", "old_value", "new_value", "reason", "confidence"}
ALLOWED_CORRECTION_FIELDS = REQUIRED_CORRECTION_FIELDS  # no extras allowed today


@dataclass
class ValidationResult:
    is_valid: bool
    parsed: Optional[List[dict]] = None
    errors: List[str] = field(default_factory=list)


def validate_response(raw_text: str, expected_row_ids: List[Any],
                       schema: DatasetSchema,
                       score_min: float = 0.0, score_max: float = 1.0,
                       allowed_decisions: Optional[List[str]] = None) -> ValidationResult:
    errors: List[str] = []

    # 1. JSON parses
    try:
        parsed = safe_json_loads(raw_text)
    except Exception as exc:
        return ValidationResult(is_valid=False, parsed=None,
                                 errors=[f"Response is not valid JSON: {exc}"])

    # 2. Top-level shape: must be a list
    if not isinstance(parsed, list):
        return ValidationResult(is_valid=False, parsed=None,
                                 errors=["Top-level response must be a JSON array."])

    # 3. Batch size must match
    if len(parsed) != len(expected_row_ids):
        errors.append(
            f"Expected {len(expected_row_ids)} rows in the response, got {len(parsed)}."
        )

    # 4. Row-level checks
    seen_ids = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            errors.append(f"Element {i} is not a JSON object.")
            continue

        if "row_id" not in item:
            errors.append(f"Element {i} is missing 'row_id'.")
            continue
        row_id = item["row_id"]
        seen_ids.append(row_id)

        if "corrections" not in item:
            errors.append(f"Row {row_id}: missing 'corrections' field.")
            continue
        corrections = item["corrections"]
        if not isinstance(corrections, list):
            errors.append(f"Row {row_id}: 'corrections' must be a list.")
            continue

        for j, corr in enumerate(corrections):
            if not isinstance(corr, dict):
                errors.append(f"Row {row_id}, correction {j}: not a JSON object.")
                continue

            keys = set(corr.keys())
            missing = REQUIRED_CORRECTION_FIELDS - keys
            extra = keys - ALLOWED_CORRECTION_FIELDS
            if missing:
                errors.append(f"Row {row_id}, correction {j}: missing fields {sorted(missing)}.")
            if extra:
                errors.append(f"Row {row_id}, correction {j}: unexpected fields {sorted(extra)}.")

            if "confidence" in corr:
                conf = corr["confidence"]
                if not isinstance(conf, (int, float)):
                    errors.append(f"Row {row_id}, correction {j}: confidence must be numeric.")
                elif not (0.0 <= float(conf) <= 1.0):
                    errors.append(f"Row {row_id}, correction {j}: confidence {conf} out of range [0,1].")

            field_name = corr.get("field")
            new_value = corr.get("new_value")

            if field_name in (schema.score_columns or []):
                if isinstance(new_value, (int, float)):
                    if not (score_min <= float(new_value) <= score_max):
                        errors.append(
                            f"Row {row_id}, correction {j}: score value {new_value} "
                            f"out of range [{score_min},{score_max}]."
                        )

            if allowed_decisions and field_name == schema.decision_column:
                if new_value not in allowed_decisions:
                    errors.append(
                        f"Row {row_id}, correction {j}: decision '{new_value}' not in "
                        f"allowed set {allowed_decisions}."
                    )

    # 5. Duplicate row_ids
    dupes = {rid for rid in seen_ids if seen_ids.count(rid) > 1}
    if dupes:
        errors.append(f"Duplicate row_ids in response: {sorted(dupes, key=str)}.")

    # 6. Missing row_ids (present in request, absent in response)
    expected_set = list(expected_row_ids)
    missing_ids = [rid for rid in expected_set if rid not in seen_ids]
    if missing_ids:
        errors.append(f"Missing row_ids in response: {missing_ids}.")

    extra_ids = [rid for rid in seen_ids if rid not in expected_set]
    if extra_ids:
        errors.append(f"Unexpected row_ids in response (not in request): {extra_ids}.")

    # 7. Ordering must match the request
    if not errors and seen_ids != expected_set:
        errors.append("Row order in response does not match the order of the request.")

    return ValidationResult(is_valid=len(errors) == 0, parsed=parsed, errors=errors)
