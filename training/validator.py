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

This file ALSO contains a second, independent check: `check_decision_plausibility`.
It is NOT part of `validate_response` and never fails/blocks a batch or
triggers the LLM repair loop, for a specific reason -- `decision` in this
dataset is a stochastic sample (see decision_plausibility.py), not a
deterministic function of the other columns. A "wrong-looking" decision is
frequently just a legitimate low-probability draw, so it would be incorrect
to treat it as a malformed LLM response the way `validate_response` treats
a missing field. Instead, `check_decision_plausibility` produces
non-blocking escalation records for a human/second-tier review queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from schema import DatasetSchema
from utils import safe_json_loads
from domain_spec import present_labels, LABEL_SCORE_BOUNDS
import decision_plausibility as dp

REQUIRED_CORRECTION_FIELDS = {"field", "old_value", "new_value", "reason", "confidence"}
ALLOWED_CORRECTION_FIELDS = REQUIRED_CORRECTION_FIELDS  # no extras allowed today


def _effective_label_columns(schema: DatasetSchema) -> List[str]:
    """The set of columns the LLM is actually permitted to correct.

    Mirrors prompts.py's own `present_labels(available) or schema.label_columns`
    logic exactly, so the prompt sent to the LLM and the validator enforcing
    its response are always looking at the same allow-list -- this is what
    prevents a correction targeting a feature column (e.g.
    `previous_trust_score`, `transaction_risk`, `fraud_history`) from being
    accepted just because it happens to exist as a dataframe column.
    """
    available = schema.column_names()
    return present_labels(available) or schema.label_columns or []


@dataclass
class ValidationResult:
    is_valid: bool
    parsed: Optional[List[dict]] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class PlausibilityReport:
    """Result of `check_decision_plausibility` for one batch.

    Never sets is_valid=False on a ValidationResult and never feeds into the
    LLM repair-prompt loop -- see module docstring above.
    """
    escalations: List[dict] = field(default_factory=list)   # band in {"borderline","implausible"}
    checked: int = 0


def validate_response(raw_text: str, expected_row_ids: List[Any],
                       schema: DatasetSchema,
                       score_min: float = 0.0, score_max: float = 1.0,
                       allowed_decisions: Optional[List[str]] = None) -> ValidationResult:
    errors: List[str] = []
    effective_labels = _effective_label_columns(schema)

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

            # Field-restriction check: the LLM may only propose corrections
            # for actual label columns (trust_score/risk_score/decision/
            # confidence, or schema.label_columns for a non-default
            # dataset), never for a feature column that merely exists in
            # the dataframe (e.g. previous_trust_score, transaction_risk,
            # fraud_history -- see examples.py Example F / the
            # NAME_COLLISION_WARNING in domain_spec.py). Previously this was
            # only prompt-text guidance with no code-level enforcement.
            if effective_labels and field_name not in effective_labels:
                errors.append(
                    f"Row {row_id}, correction {j}: field '{field_name}' is not a "
                    f"correctable label (allowed: {effective_labels}). Features are "
                    "read-only evidence, never a correction target."
                )
                continue

            if field_name in (schema.score_columns or []):
                # Prefer a per-field bound (e.g. confidence's 0.5 floor) over
                # the generic score_min/score_max when one is declared, so a
                # field-specific hard constraint from domain_spec.py is
                # actually enforced instead of only the generic [0,1] range.
                field_min, field_max = LABEL_SCORE_BOUNDS.get(field_name, (score_min, score_max))
                if isinstance(new_value, (int, float)):
                    if not (field_min <= float(new_value) <= field_max):
                        errors.append(
                            f"Row {row_id}, correction {j}: score value {new_value} "
                            f"out of range [{field_min},{field_max}] for field '{field_name}'."
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


def check_decision_plausibility(
    df_full: "Any",
    row_ids: List[Any],
    decisions: List[int],
    schema: DatasetSchema,
    ctx: Optional["dp.PlausibilityContext"] = None,
) -> PlausibilityReport:
    """Cross-check a batch's decisions against the reconstructed generator
    distribution (decision_plausibility.py).

    Args:
        df_full: the FULL input dataframe (needed so normalization/IPF
            calibration is computed dataset-wide, matching generate_labels.py).
            Ignored if `ctx` is already built and passed in -- callers doing
            many batches should build `ctx` once via `dp.build_context(df_full)`
            and pass it here to avoid recomputing it per batch.
        row_ids: the row_ids in this batch, in the schema's id_column space.
        decisions: the (possibly LLM-corrected) decision values for those rows,
            same order as row_ids.
        schema: inferred DatasetSchema, used only to sanity-check that a
            decision column actually exists.
        ctx: a pre-built PlausibilityContext (recommended for multi-batch runs).

    Returns:
        PlausibilityReport with one escalation entry per row classified as
        "borderline" or "implausible". Rows classified "ok" are omitted.
        This never raises and never marks the batch invalid -- callers decide
        what to do with escalations (log, route to a review queue, etc.).
    """
    if schema.decision_column is None:
        return PlausibilityReport(escalations=[], checked=0)

    if ctx is None:
        ctx = dp.build_context(df_full, id_column=schema.id_column)

    results = dp.evaluate_batch(ctx, row_ids, decisions)
    escalations = [r for r in results if r["band"] != "ok"]
    return PlausibilityReport(escalations=escalations, checked=len(results))