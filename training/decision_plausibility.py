"""
decision_plausibility.py
========================
Verification spec for the `decision` column produced by generate_labels.py.

WHY THIS FILE EXISTS
---------------------
`decision` is NOT a deterministic function of `trust_score` / `risk_score`.
generate_labels.py builds a per-row categorical distribution over the 4
decision classes (rule-adjusted logits -> IPF-calibrated -> softmax(temp=1.2))
and then draws the final label via Gumbel-max SAMPLING from that
distribution. Empirically, rows with near-identical (trust_score, risk_score)
routinely produce all four decisions -- e.g. trust~0.87/risk~0.25 in the
100k-row reference dataset yields 1712 ALLOW / 393 CHALLENGE / 122 OTP /
29 REJECT rows.

Consequence: an LLM (or a hard formula) auditing this dataset can NEVER
reconstruct `decision` exactly from trust/risk alone, and "doesn't match my
formula" is not evidence of an error. What we CAN check is: is the assigned
decision a plausible draw from the distribution the generator would have
produced for that row? This module reconstructs that distribution from raw
features and flags only decisions that are statistically implausible given
it -- never decisions that are merely "not the mode."

DRIFT WARNING
-------------
The formulas below are copied from generate_labels.py's `_apply_probabilistic_rules`
and `_calibrate_and_sample`. If that file's weights change, this file will
silently go stale. SOURCE_SHA256 below is the hash of generate_labels.py at
the time this module was written; `build_context()` will emit a loud warning
(not a hard failure) if the hash no longer matches, so a future formula change
doesn't fail silently.

KNOWN RECONSTRUCTION LIMITATION (persona-conditioned latents)
---------------------------------------------------------------
generate_labels.py assigns each row a stochastic "persona" via a seeded
Gumbel-max draw (`_generate_personas`), then *modulates specific latents*
before they reach `base_logits`/`_apply_probabilistic_rules` depending on
that persona -- e.g. for the "Fraudster" persona, `fraud_propensity` is
boosted (`*1.5 + 0.2`, clipped) and `session_anomaly` is boosted (`*1.5`,
clipped); for "Bot/Replay", `behavioral_anomaly` is boosted and
`voice_authenticity` is reduced (`*0.3`).

This module recomputes `fraud_propensity`/`session_anomaly` (and the other
interaction terms) directly from raw features using the SAME formulas
generate_labels.py uses *before* persona modulation, because persona
assignment depends on a seeded RNG draw this module has no access to and
cannot replay. Practically:
  - For the subset of rows the generator internally assigned to the
    "Fraudster" (or "Bot/Replay") persona, the true `base_logits[:,2]`
    (VOICE_AND_OTP) contribution from `session_anomaly` and the true
    `fraud_impact` rule shift are systematically HIGHER than what this
    module reconstructs, because it uses the pre-boost values.
  - This is a real, systematic reconstruction gap -- distinct from (and in
    addition to) the Gumbel-sampling randomness this module is already
    designed to tolerate via IMPLAUSIBLE_FLOOR/BORDERLINE_FLOOR. The
    SHA256 drift check above only detects *future* edits to
    generate_labels.py; it does not and cannot detect this gap, which has
    been present since this module was first derived.
  - Net effect: rows in the "Fraudster"/"Bot/Replay" persona subset that
    correctly received a higher-tier decision (VOICE_AND_OTP/REJECT) are
    slightly more likely to be flagged "borderline"/"implausible" by this
    module than they should be, since their true generator-side
    probability mass on that decision is underestimated here.
  - No math redesign is planned for this: exactly replaying persona
    assignment would require duplicating generate_labels.py's persona
    logit computation AND its seeded `np.random.default_rng` draw stream
    inside this module, which reintroduces the exact duplication problem
    this module's DRIFT WARNING above is trying to minimize, for a
    stochastic effect that IMPLAUSIBLE_FLOOR/BORDERLINE_FLOOR already
    exist to absorb. If tighter fidelity is ever needed, the floors can be
    widened specifically for suspected Fraudster/Bot-persona rows instead.

USAGE
-----
    from decision_plausibility import build_context, evaluate_batch

    ctx = build_context(df)                      # df = the FULL input CSV
    results = evaluate_batch(ctx, df.index, df['decision'])
    # results[i] = {"prob": ..., "band": "ok"/"borderline"/"implausible",
    #               "expected_top_decision": ..., "expected_top_prob": ...,
    #               "suggested_min_decision": ...}
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from domain_spec import DECISION_VALUES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants mirrored from generate_labels.py -- keep these two files in sync.
# ---------------------------------------------------------------------------
# NOTE: the decision name->value mapping itself is NOT duplicated here -- it
# is imported from domain_spec.py (the single source of truth also used by
# validator.py/config.py for `allowed_decisions` enforcement). Only the
# generate_labels.py-specific numeric constants below (target distribution,
# temperature, IPF settings) remain local to this file.
DECISIONS = DECISION_VALUES
DECISION_NAMES = [name for name, _ in sorted(DECISION_VALUES.items(), key=lambda kv: kv[1])]

TARGET_DISTRIBUTION = np.array([0.62, 0.20, 0.12, 0.06])
TEMPERATURE = 1.2
IPF_MAX_ITER = 50
IPF_TOL = 0.005

MAGNITUDE_FEATS = ["transaction_amount", "successful_transactions",
                   "failed_attempts", "account_age_days"]
RANK_FEATS = ["beneficiary_frequency", "fraud_history"]

REQUIRED_COLUMNS = [
    "trust_score", "risk_score", "spoof_probability", "liveness_score",
    "stress_score", "speaker_similarity", "previous_trust_score",
    "beneficiary_frequency", "transaction_amount", "account_age_days",
    "failed_attempts", "fraud_history",
]

# Default plausibility thresholds (probability the generator would have
# assigned to the decision that was actually recorded).
#
# IMPLAUSIBLE_FLOOR mirrors generate_labels.py's own internal sanity check
# (`_validate`: "impossible state" combinations must affect < 1% of rows),
# so a row scoring below this floor is inconsistent with the generator's own
# definition of a valid dataset, not just "unlikely."
IMPLAUSIBLE_FLOOR = 0.01
BORDERLINE_FLOOR = 0.05

# Per user policy: borderline/implausible rows are escalated toward higher
# verification (CHALLENGE / OTP), never auto-pushed toward REJECT and never
# auto-relaxed toward ALLOW.
_ESCALATION_FLOOR = {
    "borderline": DECISIONS["VOICE_CHALLENGE"],
    "implausible": DECISIONS["VOICE_AND_OTP"],
}

SOURCE_FILE_HINT = "generate_labels.py"
# Hash of generate_labels.py as read for this module's derivation. Populated
# by build_context() at call time against the real file when available.
SOURCE_SHA256 = "6d7102a99622de38a3a086437ac4264132994083faf4c6cebd3f6ef55abd0f0c"


def _sha256_of_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def _softmax(logits: np.ndarray, temp: float = TEMPERATURE) -> np.ndarray:
    scaled = logits / temp
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log1p_minmax(series: pd.Series) -> np.ndarray:
    val = np.log1p(series.clip(lower=0).to_numpy(dtype=float))
    vmin, vmax = val.min(), val.max()
    if vmax == vmin:
        return np.zeros_like(val)
    return (val - vmin) / (vmax - vmin)


def _rank_pct(series: pd.Series) -> np.ndarray:
    return series.rank(pct=True).to_numpy()


@dataclass
class PlausibilityContext:
    """Holds the reconstructed per-row expected decision distribution for a
    dataset, plus the thresholds used to classify observed decisions."""

    final_probs: np.ndarray                 # (n, 4)
    row_index: pd.Index                     # index alignment back to caller's df
    implausible_floor: float = IMPLAUSIBLE_FLOOR
    borderline_floor: float = BORDERLINE_FLOOR
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def _pos(self, row_id: Any) -> int:
        return self.row_index.get_loc(row_id)

    def probs_for(self, row_id: Any) -> np.ndarray:
        return self.final_probs[self._pos(row_id)]


def build_context(
    df: pd.DataFrame,
    implausible_floor: float = IMPLAUSIBLE_FLOOR,
    borderline_floor: float = BORDERLINE_FLOOR,
    generate_labels_path: Optional[Path] = None,
    id_column: Optional[str] = None,
) -> PlausibilityContext:
    """Reconstruct the expected per-row decision distribution.

    `df` must be the FULL dataset being verified (not a single batch): the
    magnitude/rank normalizations and the IPF calibration are computed
    relative to the whole dataset's distribution, exactly as
    generate_labels.py does.

    `id_column`: if the pipeline looks rows up by an id column (e.g.
    "user_id", from schema.id_column) rather than by dataframe position,
    pass it here so evaluate_row/evaluate_batch can be called with those
    same id values later. If omitted, rows are keyed by `df.index`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"build_context: missing required columns: {missing}")

    if generate_labels_path is None:
        candidate = Path(__file__).resolve().parent / SOURCE_FILE_HINT
        generate_labels_path = candidate if candidate.exists() else None
    if generate_labels_path is not None:
        current_hash = _sha256_of_file(generate_labels_path)
        if current_hash and SOURCE_SHA256 != "unknown" and current_hash != SOURCE_SHA256:
            warnings.warn(
                f"{SOURCE_FILE_HINT} hash has changed since decision_plausibility.py "
                "was derived from it. Rule weights/constants in this module may be "
                "stale -- re-derive them before trusting plausibility flags.",
                stacklevel=2,
            )

    n = len(df)
    norm: Dict[str, np.ndarray] = {}
    for f in MAGNITUDE_FEATS:
        norm[f + "_norm"] = _log1p_minmax(df[f])
    for f in RANK_FEATS:
        norm[f + "_norm"] = _rank_pct(df[f])

    trust = df["trust_score"].to_numpy(dtype=float)
    risk = df["risk_score"].to_numpy(dtype=float)

    # --- base logits (generate_labels.py: run(), lines ~437-441) ---
    base_logits = np.zeros((n, 4))
    base_logits[:, 0] = trust * 3.5 - risk * 2.0
    base_logits[:, 1] = (1.0 - trust) * 1.5 + risk * 1.0
    # NOTE: pre-persona-modulation session_anomaly -- see "KNOWN
    # RECONSTRUCTION LIMITATION" in the module docstring. generate_labels.py
    # boosts this by *1.5 (clipped) for rows assigned the "Fraudster"
    # persona before it reaches base_logits[:,2] there.
    session_anomaly = df["spoof_probability"].to_numpy(dtype=float) * 2 + \
        (1.0 - df["liveness_score"].to_numpy(dtype=float))
    base_logits[:, 2] = risk * 2.5 + session_anomaly - trust * 1.0
    base_logits[:, 3] = risk * 4.0 - trust * 2.5 - 1.5

    # --- rule-trigger proxies (generate_labels.py: _calculate_latents /
    # _calculate_nonlinear_interactions), rebuilt from raw feature columns ---
    # NOTE: these are the PRE-persona-modulation formulas. generate_labels.py
    # boosts `fraud_propensity` and `session_anomaly` further for rows it
    # internally assigned the "Fraudster" persona (and modulates other
    # latents for "Bot/Replay") -- see the "KNOWN RECONSTRUCTION LIMITATION"
    # section in this module's docstring for why that isn't replayed here.
    fraud_propensity = _sigmoid(
        norm["fraud_history_norm"] * 3 + norm["failed_attempts_norm"] * 2 - 1.0
    )
    multimodal_conflict = (
        df["speaker_similarity"].to_numpy(dtype=float)
        * df["stress_score"].to_numpy(dtype=float)
        * (1.0 - df["liveness_score"].to_numpy(dtype=float))
    )
    repeated_safe_behavior = (
        norm["beneficiary_frequency_norm"]
        * (1.0 - norm["transaction_amount_norm"])
        * df["previous_trust_score"].to_numpy(dtype=float)
    )

    # --- probabilistic rules (generate_labels.py: _apply_probabilistic_rules) ---
    logits = base_logits.copy()
    spoof_impact = _sigmoid((df["spoof_probability"].to_numpy(dtype=float) - 0.7) * 10) * 4.0
    logits[:, 3] += spoof_impact

    fraud_impact = (fraud_propensity ** 2) * 3.0
    logits[:, 3] += fraud_impact * 0.7
    logits[:, 2] += fraud_impact * 0.5

    conflict_impact = multimodal_conflict * 3.5
    logits[:, 1] += conflict_impact
    logits[:, 0] -= conflict_impact * 0.5

    safe_impact = repeated_safe_behavior * 3.0
    logits[:, 0] += safe_impact

    # --- IPF calibration to the target class distribution
    # (generate_labels.py: _calibrate_and_sample) ---
    calibrated = logits.copy()
    for _ in range(IPF_MAX_ITER):
        probs = _softmax(calibrated, temp=TEMPERATURE)
        current_dist = probs.mean(axis=0)
        if np.all(np.abs(current_dist - TARGET_DISTRIBUTION) < IPF_TOL):
            break
        ratio = TARGET_DISTRIBUTION / (current_dist + 1e-9)
        calibrated += np.log(ratio)

    final_probs = _softmax(calibrated, temp=TEMPERATURE)

    diagnostics = {
        "reconstructed_mean_distribution": final_probs.mean(axis=0).tolist(),
        "target_distribution": TARGET_DISTRIBUTION.tolist(),
    }
    if "decision" in df.columns:
        argmax_dec = final_probs.argmax(axis=1)
        actual = df["decision"].to_numpy()
        diagnostics["argmax_match_rate_vs_actual"] = float((argmax_dec == actual).mean())
        assigned_p = final_probs[np.arange(n), np.clip(actual, 0, 3)]
        diagnostics["assigned_prob_mean"] = float(assigned_p.mean())
        diagnostics["assigned_prob_below_implausible_floor"] = float(
            (assigned_p < implausible_floor).mean()
        )

    logger.info(
        "decision_plausibility.build_context: reconstructed dist=%s (target=%s)",
        np.round(final_probs.mean(axis=0), 4), TARGET_DISTRIBUTION,
    )

    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(f"build_context: id_column '{id_column}' not in dataframe")
        row_index = pd.Index(df[id_column].tolist())
        if row_index.has_duplicates:
            raise ValueError(
                f"build_context: id_column '{id_column}' has duplicate values; "
                "cannot uniquely key rows for plausibility lookup."
            )
    else:
        row_index = df.index

    return PlausibilityContext(
        final_probs=final_probs,
        row_index=row_index,
        implausible_floor=implausible_floor,
        borderline_floor=borderline_floor,
        diagnostics=diagnostics,
    )


def _classify(prob: float, ctx: PlausibilityContext) -> str:
    if prob < ctx.implausible_floor:
        return "implausible"
    if prob < ctx.borderline_floor:
        return "borderline"
    return "ok"


def _suggested_min_decision(decision: int, band: str) -> int:
    """Never suggests REJECT and never suggests relaxing toward ALLOW --
    only raises the verification tier for borderline/implausible rows."""
    floor = _ESCALATION_FLOOR.get(band)
    if floor is None:
        return decision
    return max(decision, floor) if decision < DECISIONS["REJECT"] else decision


def evaluate_row(ctx: PlausibilityContext, row_id: Any, decision: int) -> Dict[str, Any]:
    probs = ctx.probs_for(row_id)
    decision = int(decision)
    prob = float(probs[decision]) if 0 <= decision <= 3 else 0.0
    band = _classify(prob, ctx)
    top_idx = int(np.argmax(probs))
    return {
        "row_id": row_id,
        "decision": decision,
        "decision_name": DECISION_NAMES[decision] if 0 <= decision <= 3 else "INVALID",
        "prob": round(prob, 4),
        "band": band,
        "expected_top_decision": top_idx,
        "expected_top_decision_name": DECISION_NAMES[top_idx],
        "expected_top_prob": round(float(probs[top_idx]), 4),
        "full_distribution": {DECISION_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
        "suggested_min_decision": _suggested_min_decision(decision, band),
    }


def evaluate_batch(
    ctx: PlausibilityContext, row_ids: Iterable[Any], decisions: Iterable[int]
) -> List[Dict[str, Any]]:
    return [evaluate_row(ctx, rid, dec) for rid, dec in zip(row_ids, decisions)]


if __name__ == "__main__":
    # Lightweight self-test / sanity report when run directly against a CSV.
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Reconstruct decision plausibility diagnostics")
    p.add_argument("csv_path")
    args = p.parse_args()

    data = pd.read_csv(args.csv_path)
    context = build_context(data)
    print("Diagnostics:")
    for k, v in context.diagnostics.items():
        print(f"  {k}: {v}")

    if "decision" in data.columns:
        results = evaluate_batch(context, data.index, data["decision"])
        bands = pd.Series([r["band"] for r in results]).value_counts(normalize=True)
        print("\nBand distribution across dataset:")
        print(bands)
        sys.exit(0)