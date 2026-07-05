"""
training/analyze_dataset.py
============================================================
Production-grade quality analyzer for the Authentication
Network's training dataset.

PURPOSE
-------
Validates training/data/transactions.csv against the exact
Authentication Network input schema (31 model features + ID
columns) before augmentation and before model training.
Designed to be re-run after every dataset generation step.

Unlike a heuristic-only analyzer, feature identity here is
schema-driven: EXPECTED_FEATURES and ID_COLUMNS are declared
explicitly below rather than inferred from column
cardinality. This avoids misclassifying high-cardinality
integer features (e.g. successful_transactions,
beneficiary_frequency) as identifiers, which uniqueness-ratio
heuristics are prone to do.

OUTPUT
------
1. A concise console summary.
2. training/data/Dataset_Info.md -- a fully-formatted Markdown
   report with real headings and tables (not console text
   reused verbatim).

REQUIREMENTS
------------
Python 3.11+, pandas, numpy, pathlib, hashlib only. No
plotting libraries. Single file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

DATASET = Path("training/data/dataset.csv")
OUTPUT_MD = Path("training/data/Dataset_Info.md")

# VIF (multicollinearity) requires an O(features^2 x rows) pass.
# Turn off for quick iteration on wide datasets.
ENABLE_VIF = True

# --------------------------------------------------------
# Explicit schema: the Authentication Network's 31 input
# features, grouped into logical families. This is the
# single source of truth for "what should this dataset look
# like" -- edit here when the model's feature set changes.
# --------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "Identity": [
        "account_age_days",
        "kyc_verified",
        "phone_verified",
        "email_verified",
        "voice_enrolled",
    ],
    "Voice Biometrics": [
        "speaker_similarity",
        "liveness_score",
        "audio_quality",
        "spoof_probability",
    ],
    "Behavior": [
        "speech_rate_similarity",
        "pronunciation_similarity",
        "command_familiarity",
        "stress_score",
        "hesitation_score",
    ],
    "Vehicle Context": [
        "vehicle_speed",
        "engine_running",
        "location_familiarity",
        "time_familiarity",
        "driver_present",
        "seatbelt_fastened",
    ],
    "Historical Profile": [
        "previous_trust_score",
        "failed_attempts",
        "successful_transactions",
        "fraud_history",
    ],
    "Transaction": [
        "transaction_amount",
        "transaction_category",
        "beneficiary_type",
        "beneficiary_frequency",
        "transaction_risk",
    ],
    "Intent": [
        "intent_type",
        "llm_confidence",
    ],
}

# Explicit identifier columns. Any of these present in the
# dataset are treated as IDs regardless of cardinality; no
# uniqueness-ratio inference is used.
ID_COLUMNS = ["user_id", "transaction_id"]

# Columns that should NEVER be present in a pre-label feature
# dataset. Their presence signals the label-generation step
# ran upstream of this file, or that label leakage occurred.
LEAKAGE_COLUMNS = [
    "trust_score", "risk_score", "decision", "confidence",
    "label_source", "label_quality",
]

FAMILY_ORDER = ["Identifier", *FEATURE_GROUPS.keys(), "Unexpected / Non-Schema"]

EXPECTED_FEATURES = [feat for feats in FEATURE_GROUPS.values() for feat in feats]

# A numeric column with <= this many unique values is treated
# as categorical (ordinal-encoded) rather than continuous,
# unless it has only 2 values (-> binary instead).
CATEGORICAL_CARDINALITY_THRESHOLD = 15

# [0, 1]-bounded score/probability-style features get an
# automatic expected range. Add entries as the schema grows.
EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    "account_age_days": (1, 5000),
    "speaker_similarity": (0.0, 1.0),
    "liveness_score": (0.0, 1.0),
    "audio_quality": (0.0, 1.0),
    "spoof_probability": (0.0, 1.0),
    "speech_rate_similarity": (0.0, 1.0),
    "pronunciation_similarity": (0.0, 1.0),
    "command_familiarity": (0.0, 1.0),
    "stress_score": (0.0, 1.0),
    "hesitation_score": (0.0, 1.0),
    "vehicle_speed": (0, 250),
    "location_familiarity": (0.0, 1.0),
    "time_familiarity": (0.0, 1.0),
    "previous_trust_score": (0.0, 1.0),
    "failed_attempts": (0, 50),
    "successful_transactions": (0, 10000),
    "llm_confidence": (0.0, 1.0),
    "transaction_risk": (0.0, 1.0),
    "beneficiary_frequency": (0, 10000),
    "transaction_amount": (0, 10_000_000),
}

IQR_MULTIPLIER = 1.5
Z_SCORE_THRESHOLD = 3.0
HIGH_CORRELATION_THRESHOLD = 0.80
REDUNDANT_CORRELATION_THRESHOLD = 0.95
LEAKAGE_CORRELATION_THRESHOLD = 0.90
HIGH_SKEW_THRESHOLD = 2.0
HIGH_KURTOSIS_THRESHOLD = 7.0
IMBALANCE_THRESHOLD = 0.98
HEAVY_OUTLIER_PCT = 0.05
HIGH_VIF_THRESHOLD = 10.0

FeatureType = Literal["binary", "categorical", "continuous"]

# Sub-score weights for the composite health score. Must sum to 100.
HEALTH_WEIGHTS = {
    "Schema": 20,
    "Completeness": 15,
    "Distribution": 15,
    "Consistency": 15,
    "Correlation": 15,
    "Feature Quality": 20,
}


# ==========================================================
# Report buffer
# ==========================================================
class Report:
    def __init__(self) -> None:
        self.console_lines: list[str] = []
        self.md_lines: list[str] = ["# Dataset Information\n"]

    def section(self, title: str) -> None:
        self.console_lines.append("\n" + "=" * 70)
        self.console_lines.append(title.upper())
        self.console_lines.append("=" * 70)
        self.md_lines.append(f"\n## {title}\n")

    def subsection(self, title: str) -> None:
        self.console_lines.append(f"\n-- {title} --")
        self.md_lines.append(f"\n### {title}\n")

    def bullet(self, text: str) -> None:
        self.console_lines.append(f"- {text}")
        self.md_lines.append(f"- {text}")

    def line(self, text: str = "") -> None:
        self.console_lines.append(text)
        self.md_lines.append(text)

    def table(self, df: pd.DataFrame, index: bool = True) -> None:
        if df is None or df.empty:
            return
        self.console_lines.append(df.to_string(index=index))
        try:
            self.md_lines.append(df.to_markdown(index=index))
        except ImportError:
            # tabulate not installed -- fall back to a plain code block
            self.md_lines.append("```\n" + df.to_string(index=index) + "\n```")

    def flush_console(self) -> None:
        print("\n".join(self.console_lines))

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.md_lines), encoding="utf-8")


# ==========================================================
# Schema-driven column classification
# ==========================================================
def classify_columns(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, FeatureType]]:
    """Returns (family_of, feature_type_of). Family comes from the
    explicit schema (or 'Unexpected / Non-Schema' / 'Identifier');
    feature_type is inferred only for non-ID columns."""
    family_of: dict[str, str] = {}
    feature_type: dict[str, FeatureType] = {}

    feature_to_family = {
        feat: family for family, feats in FEATURE_GROUPS.items() for feat in feats
    }

    for col in df.columns:
        if col in ID_COLUMNS:
            family_of[col] = "Identifier"
            continue
        family_of[col] = feature_to_family.get(col, "Unexpected / Non-Schema")

        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 2:
            feature_type[col] = "binary"
        elif series.dtype == object or nunique <= CATEGORICAL_CARDINALITY_THRESHOLD:
            feature_type[col] = "categorical"
        else:
            feature_type[col] = "continuous"

    return family_of, feature_type


def cols_by_family(cols: list[str], family_of: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {f: [] for f in FAMILY_ORDER}
    for col in cols:
        grouped.setdefault(family_of.get(col, "Unexpected / Non-Schema"), [])
        grouped[family_of.get(col, "Unexpected / Non-Schema")].append(col)
    return grouped


def render_by_family(
    report: Report,
    section_title: str,
    cols: list[str],
    family_of: dict[str, str],
    compute_fn,
    empty_msg: str = "No applicable features in this dataset.",
) -> None:
    """Runs compute_fn(cols_subset) -> DataFrame for each family that
    has applicable columns, and renders one table per family under
    the given section."""
    report.section(section_title)
    grouped = cols_by_family(cols, family_of)
    rendered_any = False
    for family in FAMILY_ORDER:
        subset = grouped.get(family, [])
        if not subset:
            continue
        result = compute_fn(subset)
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            continue
        report.subsection(family)
        report.table(result)
        rendered_any = True
    if not rendered_any:
        report.line(empty_msg)


# ==========================================================
# Dataset fingerprint
# ==========================================================
def compute_fingerprint(path: Path) -> str:
    """MD5 of the raw file bytes -- gives every generated dataset
    a unique, exactly-reproducible identifier. Two files are
    guaranteed to have the same fingerprint iff they are
    byte-identical."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ==========================================================
# Dataset Overview
# ==========================================================
def dataset_overview(df: pd.DataFrame, fingerprint: str, family_of: dict[str, str], report: Report) -> None:
    report.section("Dataset Overview")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    family_counts = pd.Series(family_of).value_counts()

    report.bullet(f"Rows: {len(df):,}")
    report.bullet(f"Columns: {len(df.columns)}")
    report.bullet(f"Memory Usage: {memory_mb:.2f} MB")
    report.bullet(f"Dataset Fingerprint (MD5): `{fingerprint}`")
    report.line("")
    report.subsection("Columns per Family")
    table = pd.DataFrame({"Columns": family_counts})
    table = table.reindex([f for f in FAMILY_ORDER if f in table.index])
    report.table(table)


# ==========================================================
# Schema Validation
# ==========================================================
def schema_validation(df: pd.DataFrame, report: Report) -> dict:
    report.section("Schema Validation")

    actual = list(df.columns)
    expected_full = ID_COLUMNS + EXPECTED_FEATURES
    expected_set = set(expected_full)

    missing = [f for f in expected_full if f not in actual]
    extra = [c for c in actual if c not in expected_set]

    common_expected_order = [f for f in expected_full if f in actual]
    common_actual_order = [c for c in actual if c in expected_set]
    reordered = common_expected_order != common_actual_order

    report.bullet(f"Expected columns: {len(expected_full)} ({len(ID_COLUMNS)} ID + {len(EXPECTED_FEATURES)} features)")
    report.bullet(f"Actual columns: {len(actual)}")

    if not missing:
        report.bullet("No missing schema columns. ✓")
    else:
        report.bullet(f"**Missing columns ({len(missing)}):** {missing}")

    if not extra:
        report.bullet("No unexpected columns. ✓")
    else:
        report.bullet(f"**Unexpected / extra columns ({len(extra)}):** {extra}")

    if reordered:
        report.bullet(
            "Column order does not match the declared schema order "
            "(informational -- does not affect training if columns are selected by name)."
        )
    else:
        report.bullet("Column order matches the declared schema. ✓")

    return {"missing": missing, "extra": extra, "reordered": reordered}


# ==========================================================
# Label Leakage Detection
# ==========================================================
def leakage_detection(df: pd.DataFrame, continuous_cols: list[str], report: Report) -> dict:
    report.section("Label / Data Leakage Detection")

    present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    suspicious_corr: list[tuple[str, str, float]] = []

    if not present:
        report.bullet(
            "No label columns (trust_score, risk_score, decision, confidence, "
            "label_source, label_quality) found. ✓ This looks like a pre-label "
            "feature dataset."
        )
        return {"present": present, "suspicious_corr": suspicious_corr}

    report.bullet(
        f"**Label columns found in this file: {present}.** If this is meant to be "
        "the pre-label feature dataset (transactions.csv), labels should not be "
        "present yet -- verify this file wasn't accidentally overwritten with "
        "dataset.csv, or that generate_labels.py hasn't been run upstream of this check."
    )

    numeric_label_cols = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    for label_col in numeric_label_cols:
        for feat_col in continuous_cols:
            if feat_col in present:
                continue
            corr = df[[label_col, feat_col]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= LEAKAGE_CORRELATION_THRESHOLD:
                suspicious_corr.append((feat_col, label_col, float(corr)))

    if suspicious_corr:
        report.subsection(f"Features correlating >= {LEAKAGE_CORRELATION_THRESHOLD} with a label column")
        for feat, label, corr in sorted(suspicious_corr, key=lambda x: -abs(x[2])):
            report.bullet(
                f"`{feat}` <-> `{label}`: r={corr:.3f} -- suspiciously strong; "
                "verify this feature isn't derived from or leaking the label."
            )
    else:
        report.bullet("No features show suspiciously high correlation with label columns.")

    return {"present": present, "suspicious_corr": suspicious_corr}


# ==========================================================
# Missing Values / Completeness
# ==========================================================
def missing_values(df: pd.DataFrame, family_of: dict[str, str], report: Report) -> dict[str, float]:
    def compute(cols: list[str]) -> Optional[pd.DataFrame]:
        missing = df[cols].isna().sum()
        if missing.sum() == 0:
            return None
        pct_missing = (missing / len(df)) * 100
        pct_complete = 100 - pct_missing
        table = pd.DataFrame({
            "Missing": missing,
            "Missing %": pct_missing.round(2),
            "Completeness %": pct_complete.round(2),
        })
        return table[table["Missing"] > 0]

    render_by_family(
        report, "Missing Values & Completeness", list(df.columns), family_of, compute,
        empty_msg="No missing values anywhere in the dataset. ✓",
    )

    missing = df.isna().sum()
    percent = (missing / len(df)) * 100
    return percent[percent > 0].to_dict()


# ==========================================================
# Duplicate Analysis
# ==========================================================
def duplicate_analysis(df: pd.DataFrame, id_cols_present: list[str], report: Report) -> float:
    report.section("Duplicate Analysis")

    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / len(df)) * 100 if len(df) else 0.0
    report.bullet(f"Duplicate rows (all columns identical): {dup_rows:,} ({dup_pct:.2f}%)")

    for col in id_cols_present:
        dup_ids = int(df[col].duplicated().sum())
        report.bullet(f"Duplicate `{col}` values: {dup_ids:,}")

    return dup_pct


# ==========================================================
# Summary Statistics / Distribution / Percentiles (continuous)
# ==========================================================
def summary_statistics(df: pd.DataFrame, continuous_cols: list[str], family_of: dict[str, str], report: Report) -> pd.DataFrame:
    def compute(cols: list[str]) -> pd.DataFrame:
        numeric = df[cols]

        def safe_mode(s: pd.Series):
            m = s.mode(dropna=True)
            return m.iloc[0] if not m.empty else np.nan

        return pd.DataFrame({
            "mean": numeric.mean(),
            "median": numeric.median(),
            "mode": numeric.apply(safe_mode),
            "std": numeric.std(),
            "variance": numeric.var(),
            "cv": (numeric.std() / numeric.mean()).replace([np.inf, -np.inf], np.nan),
            "min": numeric.min(),
            "max": numeric.max(),
            "range": numeric.max() - numeric.min(),
            "unique": numeric.nunique(),
        }).round(4)

    render_by_family(report, "Summary Statistics (Continuous Features)", continuous_cols, family_of, compute)

    if not continuous_cols:
        return pd.DataFrame()
    return compute(continuous_cols)


def distribution_analysis(df: pd.DataFrame, continuous_cols: list[str], family_of: dict[str, str], report: Report) -> pd.DataFrame:
    def compute(cols: list[str]) -> pd.DataFrame:
        numeric = df[cols]
        return pd.DataFrame({
            "skewness": numeric.skew(),
            "kurtosis": numeric.kurt(),
        }).round(4)

    render_by_family(report, "Distribution Analysis (Continuous Features)", continuous_cols, family_of, compute)

    if not continuous_cols:
        return pd.DataFrame()
    return compute(continuous_cols)


def percentile_analysis(df: pd.DataFrame, continuous_cols: list[str], family_of: dict[str, str], report: Report) -> None:
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    def compute(cols: list[str]) -> pd.DataFrame:
        numeric = df[cols]
        table = numeric.quantile(quantiles).T.round(4)
        table.columns = [f"p{int(q * 100)}" for q in quantiles]
        return table

    render_by_family(report, "Percentiles (Continuous Features)", continuous_cols, family_of, compute)


# ==========================================================
# Feature Range Validation
# ==========================================================
def range_validation(df: pd.DataFrame, family_of: dict[str, str], report: Report) -> dict[str, int]:
    violations: dict[str, int] = {}
    configured_cols = [c for c in EXPECTED_RANGES if c in df.columns]

    def compute(cols: list[str]) -> Optional[pd.DataFrame]:
        rows = []
        for col in cols:
            if col not in EXPECTED_RANGES:
                continue
            low, high = EXPECTED_RANGES[col]
            invalid = int(((df[col] < low) | (df[col] > high)).sum())
            violations[col] = invalid
            rows.append({
                "Feature": col, "Range": f"[{low}, {high}]",
                "Status": "PASS" if invalid == 0 else "FAIL", "Invalid": invalid,
            })
        return pd.DataFrame(rows).set_index("Feature") if rows else None

    render_by_family(
        report, "Feature Range Validation", configured_cols, family_of, compute,
        empty_msg="No configured expected ranges matched columns in this dataset.",
    )
    return violations


# ==========================================================
# Binary Feature Analysis
# ==========================================================
def binary_analysis(df: pd.DataFrame, binary_cols: list[str], family_of: dict[str, str], report: Report) -> dict[str, float]:
    imbalance: dict[str, float] = {}

    def compute(cols: list[str]) -> pd.DataFrame:
        rows = []
        for col in cols:
            counts = df[col].value_counts(dropna=True)
            percent = df[col].value_counts(normalize=True, dropna=True) * 100
            if not percent.empty:
                imbalance[col] = float(percent.max())
            values_str = "; ".join(f"{v}={counts[v]:,} ({percent[v]:.1f}%)" for v in counts.index)
            rows.append({"Feature": col, "Distribution": values_str})
        return pd.DataFrame(rows).set_index("Feature") if rows else pd.DataFrame()

    render_by_family(
        report, "Binary Feature Analysis", binary_cols, family_of, compute,
        empty_msg="No binary features detected.",
    )
    return imbalance


# ==========================================================
# Categorical Feature Analysis (with entropy)
# ==========================================================
def shannon_entropy(counts: pd.Series) -> tuple[float, float]:
    """Returns (entropy_bits, normalized_entropy in [0,1])."""
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = float(np.log2(len(counts))) if len(counts) > 1 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return entropy, normalized


def categorical_analysis(df: pd.DataFrame, categorical_cols: list[str], family_of: dict[str, str], report: Report) -> dict[str, float]:
    concentration: dict[str, float] = {}  # top-category share, for recommendations

    def compute(cols: list[str]) -> pd.DataFrame:
        rows = []
        for col in cols:
            counts = df[col].value_counts(dropna=True)
            if counts.empty:
                continue
            entropy, normalized = shannon_entropy(counts)
            top_share = float(counts.iloc[0] / counts.sum() * 100)
            concentration[col] = top_share
            rows.append({
                "Feature": col,
                "Categories": len(counts),
                "Top Category": counts.index[0],
                "Top Share %": round(top_share, 2),
                "Entropy (bits)": round(entropy, 3),
                "Normalized Entropy": round(normalized, 3),
            })
        return pd.DataFrame(rows).set_index("Feature") if rows else pd.DataFrame()

    render_by_family(
        report, "Categorical Feature Analysis", categorical_cols, family_of, compute,
        empty_msg="No categorical features detected.",
    )

    # Detailed top-10 breakdown per feature, still family-grouped.
    for col in categorical_cols:
        nunique = df[col].nunique(dropna=True)
        counts = df[col].value_counts(dropna=True).head(10)
        percent = df[col].value_counts(normalize=True, dropna=True).head(10) * 100
        report.subsection(f"{col} -- Top Categories")
        for value in counts.index:
            report.bullet(f"{value}: {counts[value]:,} ({percent[value]:.2f}%)")
        if nunique > 10:
            report.bullet(f"... and {nunique - 10} more categories")

    return concentration


# ==========================================================
# Cardinality
# ==========================================================
def cardinality_report(df: pd.DataFrame, family_of: dict[str, str], feature_type: dict[str, str], report: Report) -> None:
    def compute(cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "Unique Values": df[cols].nunique(),
            "Type": [feature_type.get(c, "id") for c in cols],
        })

    render_by_family(report, "Cardinality", list(df.columns), family_of, compute)


# ==========================================================
# Correlation Analysis + Multicollinearity
# ==========================================================
def correlation_analysis(
    df: pd.DataFrame, continuous_cols: list[str], report: Report
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    report.section("Correlation Analysis")

    if len(continuous_cols) < 2:
        report.bullet("Fewer than two continuous features -- skipping correlation analysis.")
        return [], []

    corr = df[continuous_cols].corr(method="pearson")
    cols = corr.columns
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))

    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top_positive = [p for p in pairs_sorted if p[2] > 0][:10]
    top_negative = sorted([p for p in pairs if p[2] < 0], key=lambda x: x[2])[:10]
    high_corr = [p for p in pairs if abs(p[2]) >= HIGH_CORRELATION_THRESHOLD]
    redundant = [p for p in pairs if abs(p[2]) >= REDUNDANT_CORRELATION_THRESHOLD]

    def pairs_table(pair_list, label):
        if not pair_list:
            return None
        return pd.DataFrame(pair_list, columns=["Feature A", "Feature B", label])

    report.subsection("Top Positive Correlations")
    t = pairs_table(top_positive, "r")
    report.table(t, index=False) if t is not None else report.line("None found.")

    report.subsection("Top Negative Correlations")
    t = pairs_table(top_negative, "r")
    report.table(t, index=False) if t is not None else report.line("None found.")

    report.subsection(f"Highly Correlated Pairs (|r| >= {HIGH_CORRELATION_THRESHOLD})")
    t = pairs_table(high_corr, "r")
    report.table(t, index=False) if t is not None else report.line("None found.")

    report.subsection(f"Redundant Features (|r| >= {REDUNDANT_CORRELATION_THRESHOLD})")
    t = pairs_table(redundant, "r")
    report.table(t, index=False) if t is not None else report.line("None found.")

    return high_corr, redundant


def compute_vif(df: pd.DataFrame, continuous_cols: list[str]) -> pd.DataFrame:
    """Variance Inflation Factor via plain OLS (numpy lstsq) --
    no statsmodels/sklearn dependency."""
    if len(continuous_cols) < 2:
        return pd.DataFrame(columns=["VIF"])

    data = df[continuous_cols].dropna()
    if len(data) < len(continuous_cols) + 1:
        return pd.DataFrame(columns=["VIF"])

    vifs = {}
    X_full = data.to_numpy(dtype=float)
    for i, col in enumerate(continuous_cols):
        y = X_full[:, i]
        others = np.delete(X_full, i, axis=1)
        others_with_intercept = np.column_stack([np.ones(len(others)), others])
        coeffs, *_ = np.linalg.lstsq(others_with_intercept, y, rcond=None)
        y_pred = others_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs[col] = vif

    return pd.DataFrame({"VIF": vifs}).sort_values("VIF", ascending=False)


def multicollinearity_report(df: pd.DataFrame, continuous_cols: list[str], report: Report) -> pd.DataFrame:
    report.section("Multicollinearity (VIF)")

    if not ENABLE_VIF:
        report.bullet("VIF computation disabled (ENABLE_VIF=False) -- skipped for performance.")
        return pd.DataFrame()

    vif_table = compute_vif(df, continuous_cols)
    if vif_table.empty:
        report.bullet("Not enough continuous features/rows to compute VIF.")
        return vif_table

    report.table(vif_table.round(2))
    report.line(
        "\nRule of thumb: VIF > 10 suggests problematic multicollinearity; "
        "VIF > 5 is worth investigating."
    )
    return vif_table


# ==========================================================
# Outlier Analysis (continuous features only)
# ==========================================================
def outlier_analysis(df: pd.DataFrame, continuous_cols: list[str], family_of: dict[str, str], report: Report) -> dict[str, float]:
    outlier_pct: dict[str, float] = {}
    n = len(df)

    def compute(cols: list[str]) -> pd.DataFrame:
        rows = []
        for col in cols:
            x = df[col].dropna()
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
            iqr_outliers = int(((x < lower) | (x > upper)).sum())

            std = x.std()
            z_outliers = 0 if (std == 0 or pd.isna(std)) else int(((x - x.mean()).abs() / std > Z_SCORE_THRESHOLD).sum())

            pct = (max(iqr_outliers, z_outliers) / n * 100) if n else 0.0
            outlier_pct[col] = pct
            rows.append({"Feature": col, "IQR_Outliers": iqr_outliers, "Z_Outliers": z_outliers, "Outlier %": round(pct, 2)})
        return pd.DataFrame(rows).set_index("Feature") if rows else pd.DataFrame()

    render_by_family(
        report, "Outlier Analysis (Continuous Features)", continuous_cols, family_of, compute,
        empty_msg="No continuous features detected.",
    )
    return outlier_pct


# ==========================================================
# Feature Quality Score
# ==========================================================
def feature_quality_score(
    df: pd.DataFrame,
    feature_type: dict[str, str],
    missing_pct: dict[str, float],
    range_violations: dict[str, int],
    outlier_pct: dict[str, float],
    skew_kurt: pd.DataFrame,
    binary_imbalance: dict[str, float],
    concentration: dict[str, float],
    family_of: dict[str, str],
    report: Report,
) -> pd.DataFrame:
    scores: dict[str, float] = {}

    for col in df.columns:
        if col in ID_COLUMNS:
            continue

        score = 100.0
        nunique = df[col].nunique(dropna=True)

        if missing_pct.get(col, 0) > 0:
            score -= min(missing_pct[col], 30)

        if nunique == 1:
            score -= 50
        elif nunique < 5 and feature_type.get(col) == "continuous":
            score -= 10

        if range_violations.get(col, 0) > 0:
            score -= 20

        if outlier_pct.get(col, 0) > HEAVY_OUTLIER_PCT * 100:
            score -= 15

        if not skew_kurt.empty and col in skew_kurt.index:
            if abs(skew_kurt.loc[col, "skewness"]) > HIGH_SKEW_THRESHOLD:
                score -= 10
            if abs(skew_kurt.loc[col, "kurtosis"]) > HIGH_KURTOSIS_THRESHOLD:
                score -= 10

        if binary_imbalance.get(col, 0) / 100 > IMBALANCE_THRESHOLD:
            score -= 15

        if concentration.get(col, 0) > 95:
            score -= 15  # over-concentrated categorical (near-constant)

        scores[col] = max(round(score, 1), 0.0)

    table_full = pd.DataFrame({"Quality": scores})

    def compute(cols: list[str]) -> pd.DataFrame:
        subset = table_full.loc[table_full.index.intersection(cols)]
        return subset.sort_values("Quality", ascending=False)

    render_by_family(report, "Feature Quality", list(scores.keys()), family_of, compute)

    return table_full


# ==========================================================
# Dynamic Recommendations
# ==========================================================
def dynamic_recommendations(
    df: pd.DataFrame,
    schema_result: dict,
    leakage_result: dict,
    missing_pct: dict[str, float],
    dup_pct: float,
    range_violations: dict[str, int],
    redundant_pairs: list[tuple[str, str, float]],
    high_corr_pairs: list[tuple[str, str, float]],
    skew_kurt: pd.DataFrame,
    binary_imbalance: dict[str, float],
    outlier_pct: dict[str, float],
    concentration: dict[str, float],
    vif_table: pd.DataFrame,
    report: Report,
) -> list[str]:
    report.section("Recommendations")
    notes: list[str] = []

    if schema_result["missing"]:
        notes.append(f"Schema: add missing required columns: {schema_result['missing']}.")
    if schema_result["extra"]:
        notes.append(
            f"Schema: {schema_result['extra']} are not part of the declared model "
            "schema -- confirm they're intentional (e.g. metadata) before training."
        )
    if schema_result["reordered"]:
        notes.append("Schema: column order differs from the declared schema (low risk if you select by name).")

    if leakage_result["present"]:
        notes.append(
            f"Leakage: label columns {leakage_result['present']} are present in this "
            "file -- exclude them from model input features."
        )
    for feat, label, corr in leakage_result["suspicious_corr"]:
        notes.append(f"Leakage: `{feat}` correlates {corr:.2f} with `{label}` -- investigate before training.")

    if missing_pct:
        worst = sorted(missing_pct.items(), key=lambda x: -x[1])[:5]
        notes.append(
            "Completeness: address missing values in "
            + ", ".join(f"{c} ({p:.1f}%)" for c, p in worst)
            + ("..." if len(missing_pct) > 5 else ".")
        )

    if dup_pct > 0:
        notes.append(f"Consistency: remove {dup_pct:.2f}% duplicate rows before training.")

    violated = [c for c, v in range_violations.items() if v > 0]
    if violated:
        notes.append(f"Consistency: {violated} contain out-of-range values -- clip or investigate the generator.")

    constant = [c for c in df.columns if c not in ID_COLUMNS and df[c].nunique(dropna=True) == 1]
    if constant:
        notes.append(f"Consistency: drop constant feature(s) {constant} -- they carry no signal.")

    if not skew_kurt.empty:
        skewed = skew_kurt[skew_kurt["skewness"].abs() > HIGH_SKEW_THRESHOLD]
        for col, row in skewed.iterrows():
            notes.append(f"Distribution: `{col}` has skewness {row['skewness']:.2f} -- consider a log/Box-Cox transform.")
        heavy_tailed = skew_kurt[skew_kurt["kurtosis"].abs() > HIGH_KURTOSIS_THRESHOLD]
        for col, row in heavy_tailed.iterrows():
            notes.append(f"Distribution: `{col}` has kurtosis {row['kurtosis']:.2f} -- heavy tails, check for extreme outliers.")

    imbalanced = [c for c, v in binary_imbalance.items() if v / 100 > IMBALANCE_THRESHOLD]
    if imbalanced:
        notes.append(f"Consistency: binary feature(s) {imbalanced} are >98% one class -- consider class weighting or resampling.")

    over_concentrated = [c for c, v in concentration.items() if v > 95]
    if over_concentrated:
        notes.append(f"Distribution: categorical feature(s) {over_concentrated} are >95% one category -- low information content.")

    heavy_outliers = [c for c, v in outlier_pct.items() if v > HEAVY_OUTLIER_PCT * 100]
    if heavy_outliers:
        notes.append(f"Distribution: {heavy_outliers} have >{HEAVY_OUTLIER_PCT*100:.0f}% flagged outliers -- verify these are real, not generator bugs.")

    if redundant_pairs:
        notes.append(
            "Correlation: near-duplicate feature pairs "
            + ", ".join(f"{a}/{b} (r={c:.2f})" for a, b, c in redundant_pairs)
            + " -- consider dropping one from each pair."
        )
    elif high_corr_pairs:
        notes.append(
            "Correlation: some feature pairs are strongly correlated "
            + ", ".join(f"{a}/{b} (r={c:.2f})" for a, b, c in high_corr_pairs[:5])
            + " -- not necessarily a problem, but worth being aware of for interpretability."
        )

    if not vif_table.empty and "VIF" in vif_table.columns:
        high_vif = vif_table[vif_table["VIF"] > HIGH_VIF_THRESHOLD]
        if not high_vif.empty:
            notes.append(f"Correlation: {list(high_vif.index)} have VIF > {HIGH_VIF_THRESHOLD:.0f} -- multicollinearity risk for linear models.")

    if not notes:
        notes.append("No issues detected -- dataset looks ready for training. ✓")

    for note in notes:
        report.bullet(note)

    return notes


# ==========================================================
# Weighted Health Score
# ==========================================================
def weighted_health_score(
    df: pd.DataFrame,
    schema_result: dict,
    leakage_result: dict,
    missing_pct: dict[str, float],
    dup_pct: float,
    range_violations: dict[str, int],
    redundant_pairs: list,
    high_corr_pairs: list,
    skew_kurt: pd.DataFrame,
    binary_imbalance: dict[str, float],
    outlier_pct: dict[str, float],
    vif_table: pd.DataFrame,
    feature_quality_table: pd.DataFrame,
    continuous_cols: list[str],
    report: Report,
) -> float:
    report.section("Dataset Health Score")
    breakdown_rows = []

    # ---- Schema sub-score ----
    schema_score = 100.0
    schema_score -= min(len(schema_result["missing"]) * 10, 60)
    schema_score -= min(len(schema_result["extra"]) * 5, 20)
    if schema_result["reordered"]:
        schema_score -= 5
    if leakage_result["present"]:
        schema_score -= 20
    schema_score = max(schema_score, 0.0)

    # ---- Completeness sub-score ----
    total_cells = len(df) * len(df.columns)
    total_missing = sum(
        (missing_pct.get(c, 0) / 100) * len(df) for c in missing_pct
    )
    overall_missing_pct = (total_missing / total_cells * 100) if total_cells else 0.0
    completeness_score = max(100 - overall_missing_pct * 3, 0.0)

    # ---- Distribution sub-score ----
    distribution_score = 100.0
    if not skew_kurt.empty:
        n_total = len(skew_kurt)
        n_skew = int((skew_kurt["skewness"].abs() > HIGH_SKEW_THRESHOLD).sum())
        n_kurt = int((skew_kurt["kurtosis"].abs() > HIGH_KURTOSIS_THRESHOLD).sum())
        distribution_score -= (n_skew / n_total) * 50 if n_total else 0
        distribution_score -= (n_kurt / n_total) * 50 if n_total else 0
    distribution_score = max(distribution_score, 0.0)

    # ---- Consistency sub-score ----
    consistency_score = 100.0
    consistency_score -= min(dup_pct * 2, 15)
    n_violations = sum(1 for v in range_violations.values() if v > 0)
    consistency_score -= min(n_violations * 5, 20)
    n_constant = sum(1 for c in df.columns if c not in ID_COLUMNS and df[c].nunique(dropna=True) == 1)
    consistency_score -= min(n_constant * 10, 30)
    n_imbalanced = sum(1 for v in binary_imbalance.values() if v / 100 > IMBALANCE_THRESHOLD)
    consistency_score -= min(n_imbalanced * 5, 15)
    n_outlier_heavy = sum(1 for v in outlier_pct.values() if v > HEAVY_OUTLIER_PCT * 100)
    consistency_score -= min(n_outlier_heavy * 3, 20)
    consistency_score = max(consistency_score, 0.0)

    # ---- Correlation sub-score ----
    correlation_score = 100.0
    correlation_score -= min(len(redundant_pairs) * 5, 30)
    correlation_score -= min(len(high_corr_pairs) * 2, 20)
    if ENABLE_VIF and not vif_table.empty and "VIF" in vif_table.columns:
        n_high_vif = int((vif_table["VIF"] > HIGH_VIF_THRESHOLD).sum())
        correlation_score -= min(n_high_vif * 5, 30)
    correlation_score = max(correlation_score, 0.0)

    # ---- Feature Quality sub-score ----
    relevant_quality = feature_quality_table[feature_quality_table.index.isin(
        [c for c in df.columns if c not in ID_COLUMNS]
    )]
    feature_quality_score_avg = float(relevant_quality["Quality"].mean()) if not relevant_quality.empty else 100.0

    sub_scores = {
        "Schema": schema_score,
        "Completeness": completeness_score,
        "Distribution": distribution_score,
        "Consistency": consistency_score,
        "Correlation": correlation_score,
        "Feature Quality": feature_quality_score_avg,
    }

    overall = sum(sub_scores[k] * HEALTH_WEIGHTS[k] / 100 for k in HEALTH_WEIGHTS)
    overall = round(overall, 1)

    for name, weight in HEALTH_WEIGHTS.items():
        breakdown_rows.append({
            "Sub-score": name,
            "Weight %": weight,
            "Score /100": round(sub_scores[name], 1),
            "Weighted Contribution": round(sub_scores[name] * weight / 100, 2),
        })

    report.table(pd.DataFrame(breakdown_rows).set_index("Sub-score"))

    if overall >= 95:
        status = "Excellent"
    elif overall >= 85:
        status = "Good"
    elif overall >= 70:
        status = "Fair"
    else:
        status = "Poor"

    report.line("")
    report.bullet(f"**Overall Weighted Health Score: {overall}/100 ({status})**")

    return overall


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    fingerprint = compute_fingerprint(DATASET)
    family_of, feature_type = classify_columns(df)

    id_cols_present = [c for c in ID_COLUMNS if c in df.columns]
    binary_cols = [c for c, t in feature_type.items() if t == "binary"]
    categorical_cols = [c for c, t in feature_type.items() if t == "categorical"]
    continuous_cols = [c for c, t in feature_type.items() if t == "continuous"]

    report = Report()

    dataset_overview(df, fingerprint, family_of, report)
    schema_result = schema_validation(df, report)
    leakage_result = leakage_detection(df, continuous_cols, report)
    missing_pct = missing_values(df, family_of, report)
    dup_pct = duplicate_analysis(df, id_cols_present, report)
    summary_statistics(df, continuous_cols, family_of, report)
    skew_kurt = distribution_analysis(df, continuous_cols, family_of, report)
    percentile_analysis(df, continuous_cols, family_of, report)
    range_violations = range_validation(df, family_of, report)
    binary_imbalance = binary_analysis(df, binary_cols, family_of, report)
    concentration = categorical_analysis(df, categorical_cols, family_of, report)
    cardinality_report(df, family_of, feature_type, report)
    high_corr_pairs, redundant_pairs = correlation_analysis(df, continuous_cols, report)
    vif_table = multicollinearity_report(df, continuous_cols, report)
    outlier_pct = outlier_analysis(df, continuous_cols, family_of, report)
    feature_quality_table = feature_quality_score(
        df, feature_type, missing_pct, range_violations, outlier_pct,
        skew_kurt, binary_imbalance, concentration, family_of, report,
    )
    dynamic_recommendations(
        df, schema_result, leakage_result, missing_pct, dup_pct, range_violations,
        redundant_pairs, high_corr_pairs, skew_kurt, binary_imbalance,
        outlier_pct, concentration, vif_table, report,
    )
    health_score = weighted_health_score(
        df, schema_result, leakage_result, missing_pct, dup_pct, range_violations,
        redundant_pairs, high_corr_pairs, skew_kurt, binary_imbalance, outlier_pct,
        vif_table, feature_quality_table, continuous_cols, report,
    )

    report.flush_console()
    report.write_markdown(OUTPUT_MD)

    print("\n" + "=" * 70)
    print(f"Fingerprint  : {fingerprint}")
    print(f"Health Score : {health_score}/100")
    print(f"Report saved : {OUTPUT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()