#!/usr/bin/env python3
"""
training/analyze_fraud.py
==========================

Fraud-focused dataset validation and reporting.

Complements `analyze_dataset.py` (which validates the model-facing feature
schema) with a report about the fraud simulation layer itself: has every
scenario actually fired, does the realized population match the
configured targets, and does the fraud evidence actually show up as
separation in the relevant features and in the final trust/risk/decision
labels.

Inputs
------
    training/data/fraud_context.csv   -- ground truth written by
                                          generate_users.py
    training/data/dataset.csv         -- final labeled dataset written by
                                          generate_labels.py

Output
------
    training/data/Fraud_Info.md

Usage
-----
    python training/analyze_fraud.py
    python training/analyze_fraud.py --fraud-context path.csv --dataset path.csv --output path.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from generators.fraud import FraudScenario, FraudTier, SCENARIOS, SUBSYSTEMS


# ==========================================================
# Configuration
# ==========================================================

DEFAULT_FRAUD_CONTEXT = Path("training/data/fraud_context.csv")
DEFAULT_DATASET = Path("training/data/dataset.csv")
DEFAULT_OUTPUT = Path("training/data/Fraud_Info.md")

# Features worth reporting a genuine-vs-fraud comparison for. These are
# exactly the features each ScenarioSpec is allowed to move.
EVIDENCE_FEATURES = {
    "voice": ["speaker_similarity", "liveness_score", "audio_quality", "spoof_probability"],
    "behavior": ["speech_rate_similarity", "pronunciation_similarity", "command_familiarity",
                 "stress_score", "hesitation_score"],
    "vehicle": ["vehicle_speed", "location_familiarity", "time_familiarity"],
    "history": ["previous_trust_score", "failed_attempts", "successful_transactions"],
    "transaction": ["transaction_amount", "transaction_risk", "beneficiary_frequency"],
    "intent": ["llm_confidence"],
}

DECISION_LABELS = {0: "ALLOW", 1: "VOICE_CHALLENGE", 2: "VOICE_AND_OTP", 3: "REJECT"}


def _df_to_markdown(df: pd.DataFrame, index_name: str = "") -> str:
    """Minimal DataFrame -> GitHub-flavored markdown table, with no extra
    dependency on `tabulate` (which isn't in requirements.txt)."""
    headers = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.tolist()
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==========================================================
# Report sections
# ==========================================================

def section_scenario_counts(fraud_df: pd.DataFrame) -> List[str]:
    lines = ["## Fraud Scenario Counts\n"]
    n = len(fraud_df)
    counts = fraud_df["scenario"].value_counts()

    lines.append("| Scenario | Tier | Count | % of Population |")
    lines.append("|---|---|---|---|")
    for scenario in FraudScenario:
        tier = SCENARIOS[scenario].tier.value
        c = int(counts.get(scenario.value, 0))
        lines.append(f"| {scenario.value} | {tier} | {c:,} | {c / n:.2%} |")

    missing = [
        s.value for s in FraudScenario
        if s is not FraudScenario.GENUINE and counts.get(s.value, 0) == 0
    ]
    lines.append("")
    if missing:
        lines.append(
            f"**WARNING: {len(missing)} scenario(s) never fired in this "
            f"sample: {missing}. Increase --num-users or fraud_rate.**"
        )
    else:
        lines.append("All non-genuine scenarios fired at least once. ✓")
    lines.append("")
    return lines


def section_tier_balance(fraud_df: pd.DataFrame, config_dict: dict) -> List[str]:
    lines = ["## Tier / Fraud Balance vs. Configuration\n"]
    n = len(fraud_df)
    tier_counts = fraud_df["tier"].value_counts(normalize=True)

    fraud_rate_actual = 1.0 - tier_counts.get(FraudTier.GENUINE.value, 0.0)
    lines.append(f"- Configured `fraud_rate`: {config_dict.get('fraud_rate', 'n/a')}")
    lines.append(f"- Actual fraud rate (non-genuine / total): {fraud_rate_actual:.2%}")
    lines.append("")
    lines.append("| Tier | Actual % |")
    lines.append("|---|---|")
    for tier in FraudTier:
        lines.append(f"| {tier.value} | {tier_counts.get(tier.value, 0.0):.2%} |")
    lines.append("")
    return lines


def section_feature_comparison(merged: pd.DataFrame) -> List[str]:
    lines = ["## Fraud vs. Genuine Feature Comparison\n"]
    lines.append(
        "For each subsystem's evidence features: mean value for genuine "
        "rows vs. every fraud scenario. A scenario should only show large "
        "deviation on the subsystems it's designed to affect.\n"
    )

    for subsystem, features in EVIDENCE_FEATURES.items():
        lines.append(f"### {subsystem.title()}\n")
        pivot = merged.groupby("scenario")[features].mean()
        # order rows: genuine first, then alphabetical
        order = ["genuine"] + sorted(s for s in pivot.index if s != "genuine")
        pivot = pivot.reindex(order)
        lines.append(_df_to_markdown(pivot.round(4), index_name="scenario"))
        lines.append("")
    return lines


def section_spoof_liveness_distribution(merged: pd.DataFrame) -> List[str]:
    lines = ["## Spoof Probability & Liveness Distribution (fraud vs. genuine)\n"]
    is_fraud = merged["scenario"] != "genuine"

    for col in ("spoof_probability", "liveness_score"):
        g = merged.loc[~is_fraud, col]
        f = merged.loc[is_fraud, col]
        lines.append(f"**{col}**")
        lines.append("")
        lines.append("| Population | mean | p50 | p90 | p99 | max |")
        lines.append("|---|---|---|---|---|---|")
        for name, series in (("genuine", g), ("fraud (any scenario)", f)):
            if len(series) == 0:
                continue
            q = series.quantile([0.5, 0.9, 0.99])
            lines.append(
                f"| {name} | {series.mean():.4f} | {q.loc[0.5]:.4f} | "
                f"{q.loc[0.9]:.4f} | {q.loc[0.99]:.4f} | {series.max():.4f} |"
            )
        lines.append("")
    return lines


def section_correlation(merged: pd.DataFrame) -> List[str]:
    lines = ["## Correlation: Fraud Scenario Presence vs. Trust/Risk/Decision\n"]
    is_fraud = (merged["scenario"] != "genuine").astype(int)

    rows = []
    for col in ("trust_score", "risk_score", "decision", "confidence"):
        if col in merged.columns:
            corr = np.corrcoef(is_fraud, merged[col])[0, 1]
            rows.append((col, corr))

    lines.append("| Column | Pearson r vs. is_fraud |")
    lines.append("|---|---|")
    for col, corr in rows:
        lines.append(f"| {col} | {corr:.4f} |")
    lines.append("")

    if "risk_score" in merged.columns:
        risk_corr = dict(rows).get("risk_score", 0.0)
        if risk_corr < 0.05:
            lines.append(
                "**WARNING: risk_score shows little to no positive "
                "correlation with fraud presence. Fraud injection may not "
                "be reaching the label generator.**"
            )
        else:
            lines.append(
                f"risk_score correlates positively with fraud presence "
                f"(r={risk_corr:.3f}). ✓"
            )
    lines.append("")
    return lines


def section_class_balance(merged: pd.DataFrame) -> List[str]:
    lines = ["## Decision Class Balance (overall, and fraud-conditioned)\n"]
    if "decision" not in merged.columns:
        lines.append("`decision` column not present in dataset -- skipped.\n")
        return lines

    overall = merged["decision"].map(DECISION_LABELS).value_counts(normalize=True)
    lines.append("**Overall:**\n")
    lines.append("| Decision | % |")
    lines.append("|---|---|")
    for label in DECISION_LABELS.values():
        lines.append(f"| {label} | {overall.get(label, 0.0):.2%} |")
    lines.append("")

    lines.append("**By fraud presence:**\n")
    crosstab = pd.crosstab(
        merged["scenario"] != "genuine",
        merged["decision"].map(DECISION_LABELS),
        normalize="index",
    )
    crosstab.index = ["genuine", "any_fraud"]
    lines.append(_df_to_markdown(crosstab.round(4), index_name="fraud"))
    lines.append("")
    return lines


def section_scenario_decision_breakdown(merged: pd.DataFrame) -> List[str]:
    lines = ["## Decision Breakdown Per Scenario\n"]
    if "decision" not in merged.columns:
        return lines
    crosstab = pd.crosstab(
        merged["scenario"], merged["decision"].map(DECISION_LABELS), normalize="index"
    )
    order = ["genuine"] + sorted(s for s in crosstab.index if s != "genuine")
    crosstab = crosstab.reindex(order)
    lines.append(_df_to_markdown(crosstab.round(4), index_name="scenario"))
    lines.append("")
    return lines


# ==========================================================
# Main
# ==========================================================

def build_report(fraud_df: pd.DataFrame, dataset_df: pd.DataFrame, config_dict: dict) -> str:
    merged = dataset_df.merge(fraud_df, on="user_id", how="inner")

    lines: List[str] = ["# Fraud Simulation Validation Report\n"]
    lines.append(f"- Rows in fraud_context.csv: {len(fraud_df):,}")
    lines.append(f"- Rows in dataset.csv: {len(dataset_df):,}")
    lines.append(f"- Rows joined on user_id: {len(merged):,}")
    lines.append("")

    lines += section_scenario_counts(fraud_df)
    lines += section_tier_balance(fraud_df, config_dict)
    lines += section_feature_comparison(merged)
    lines += section_spoof_liveness_distribution(merged)
    lines += section_correlation(merged)
    lines += section_class_balance(merged)
    lines += section_scenario_decision_breakdown(merged)

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraud-context", type=Path, default=DEFAULT_FRAUD_CONTEXT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.fraud_context.exists():
        raise FileNotFoundError(
            f"{args.fraud_context} not found. Run generate_users.py first."
        )
    if not args.dataset.exists():
        raise FileNotFoundError(
            f"{args.dataset} not found. Run the full generation pipeline first."
        )

    fraud_df = pd.read_csv(args.fraud_context)
    dataset_df = pd.read_csv(args.dataset)

    from generators.fraud import FraudGeneratorConfig
    config_dict = FraudGeneratorConfig().to_dict()

    report = build_report(fraud_df, dataset_df, config_dict)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)

    print(f"Fraud validation report written to {args.output}")


if __name__ == "__main__":
    main()
