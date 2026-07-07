"""
modify_dataset.py
==================

Refines `training/data/transactions.csv` into a more realistic, statistically
healthier `training/data/transactions_modified.csv` for the voice-driven
in-vehicle authentication / fraud-risk network.

This script is a REFINEMENT layer, not a redesign:
    * schema, column names, column order, dtypes, and row count are preserved
      exactly (drop-in replacement).
    * every change is small, probabilistic, and grounded in the statistical
      diagnostics captured in Dataset_Info.md (skew, kurtosis, outliers,
      correlations, binary imbalance, out-of-range values).

Design notes
------------
Distribution reshaping is done with an *order-statistic mapping*: for a
column that needs a healthier shape, we draw a synthetic sample from a
better-behaved numpy distribution, sort it, and re-assign it to the
existing rank order of the real column. Because the mapping is strictly
monotonic, every Spearman-rank relationship the column has with the rest
of the dataset (age vs. trust, trust vs. successful transactions, etc.)
is preserved automatically, while skew/kurtosis/outliers improve.

Dependency strengthening uses a *rank-copula nudge*: a column's percentile
rank is blended a small amount with the percentile rank of a "driver"
signal (e.g. account age, persona score), then mapped back through the
column's own empirical quantile function. This tightens a correlation by
a controlled amount without ever inventing values outside the column's
real empirical range.

Only pandas, numpy, and pathlib are used, and all transforms are vectorized.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # ---- paths -------------------------------------------------------
    INPUT_PATH = Path("training/data/transactions.csv")
    OUTPUT_PATH = Path("training/data/transactions_modified.csv")

    # ---- reproducibility ----------------------------------------------
    RANDOM_SEED = 42

    # ---- declared schema (order matters, must not change) -------------
    SCHEMA = [
        "user_id", "account_age_days", "kyc_verified", "phone_verified",
        "email_verified", "voice_enrolled", "speaker_similarity",
        "liveness_score", "audio_quality", "spoof_probability",
        "speech_rate_similarity", "pronunciation_similarity",
        "command_familiarity", "stress_score", "hesitation_score",
        "vehicle_speed", "engine_running", "location_familiarity",
        "time_familiarity", "driver_present", "seatbelt_fastened",
        "previous_trust_score", "failed_attempts", "successful_transactions",
        "fraud_history", "transaction_amount", "transaction_category",
        "beneficiary_type", "beneficiary_frequency", "transaction_risk",
        "intent_type", "llm_confidence",
    ]

    # ---- valid domain bounds (from Dataset_Info.md range validation) ---
    BOUNDS = {
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
        "vehicle_speed": (0.0, 250.0),
        "location_familiarity": (0.0, 1.0),
        "time_familiarity": (0.0, 1.0),
        "previous_trust_score": (0.0, 1.0),
        "failed_attempts": (0, 50),
        "successful_transactions": (0, 10000),
        "transaction_amount": (0.0, 10_000_000.0),
        "beneficiary_frequency": (0.0, 1.0),
        "transaction_risk": (0.0, 1.0),
        "llm_confidence": (0.0, 1.0),
    }

    # ---- distributions that Dataset_Info.md flagged as unhealthy -------
    # column -> (numpy sampler name, kwargs, rescale to observed [low, high])
    #
    # NOTE: `speaker_similarity` and `transaction_risk` used to be reshaped
    # here too (into [0.9123, 1.0] and [0.0, 0.60] respectively). Those
    # targets were tuned back when `generate_users.py`/`generate_transactions.py`
    # never actually produced fraudulent samples, so both columns were
    # unnaturally narrow and needed a synthetic beta/gamma stretch to look
    # "healthy". Now that fraud scenarios (see generators/fraud.py) inject
    # real low-similarity / high-risk tail samples, `reshape_distribution`'s
    # rank-preserving remap into a narrow target range would silently
    # collapse that fraud signal back into a "normal-looking" band (e.g. the
    # single lowest-similarity fraud row would still be remapped to ~0.91).
    # Do NOT add either column back here without re-deriving the target
    # range from the *post-fraud* empirical distribution.
    RESHAPE_TARGETS = {
        "account_age_days": dict(sampler="gamma", kwargs=dict(shape=1.6, scale=1.0),
                                  low=30.0, high=5000.0),
        "beneficiary_frequency": dict(sampler="beta", kwargs=dict(a=2.4, b=1.0),
                                       low=0.0, high=1.0),
    }

    # ---- dependency strengthening: target_col -> driver definition -----
    # strength in [0, 1] controls how much rank is pulled toward the driver
    DEPENDENCIES = [
        dict(target="previous_trust_score", driver="account_age_days",
             strength=0.10, direction=1),
        dict(target="successful_transactions", driver="previous_trust_score",
             strength=0.08, direction=1),
        dict(target="hesitation_score", driver="stress_score",
             strength=0.12, direction=1),
        dict(target="spoof_probability", driver="audio_quality",
             strength=0.10, direction=-1),
        dict(target="spoof_probability", driver="liveness_score",
             strength=0.10, direction=-1),
        dict(target="speaker_similarity", driver="audio_quality",
             strength=0.06, direction=1),
        dict(target="llm_confidence", driver="audio_quality",
             strength=0.10, direction=1),
        dict(target="llm_confidence", driver="stress_score",
             strength=0.06, direction=-1),
        dict(target="transaction_risk", driver="transaction_amount",
             strength=0.08, direction=1),
        dict(target="transaction_risk", driver="beneficiary_frequency",
             strength=0.10, direction=-1),
        dict(target="location_familiarity", driver="account_age_days",
             strength=0.05, direction=1),
        dict(target="time_familiarity", driver="account_age_days",
             strength=0.05, direction=1),
    ]

    # ---- noise injection -------------------------------------------------
    JITTER_FRACTION_OF_STD = 0.015   # gaussian jitter scale relative to col std
    JITTER_COLUMNS = [
        "speaker_similarity", "liveness_score", "audio_quality",
        "speech_rate_similarity", "pronunciation_similarity",
        "command_familiarity", "stress_score", "hesitation_score",
        "location_familiarity", "time_familiarity", "previous_trust_score",
        "transaction_risk", "llm_confidence",
    ]

    # ---- binary imbalance nudges (column -> max flip fraction) ---------
    BINARY_REFINEMENTS = {
        "phone_verified": 0.006,
        "driver_present": 0.006,
        "fraud_history": 0.004,
        "seatbelt_fastened": 0.010,
    }

    # ---- edge case injection rates --------------------------------------
    EDGE_CASE_FRACTION = 0.015  # per edge-case pattern


# ============================================================================
# UTILITIES
# ============================================================================

def _clip(series: pd.Series, col: str) -> pd.Series:
    """Clip a series to its declared valid domain, if one is configured."""
    bounds = Config.BOUNDS.get(col)
    if bounds is None:
        return series
    lo, hi = bounds
    return series.clip(lower=lo, upper=hi)


def _rank_pct(series: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Percentile rank in (0, 1), with tiny jitter to break ties smoothly."""
    n = len(series)
    r = series.rank(method="first").to_numpy(dtype=float)
    r = r + rng.uniform(-0.25, 0.25, size=n)
    r = np.clip(r, 1, n)
    return r / (n + 1)


def draw_sampler(name: str, kwargs: dict, size: int, rng: np.random.Generator) -> np.ndarray:
    """Thin dispatcher over numpy's Generator sampling distributions."""
    if name == "gamma":
        return rng.gamma(shape=kwargs["shape"], scale=kwargs.get("scale", 1.0), size=size)
    if name == "beta":
        return rng.beta(a=kwargs["a"], b=kwargs["b"], size=size)
    if name == "lognormal":
        return rng.lognormal(mean=kwargs.get("mean", 0.0), sigma=kwargs.get("sigma", 1.0), size=size)
    raise ValueError(f"Unknown sampler: {name}")


# ============================================================================
# PERSONA GENERATION (latent, never written to disk)
# ============================================================================

PERSONA_NAMES = [
    "trusted_commuter", "corporate_employee", "frequent_traveler",
    "new_customer", "senior_citizen", "high_value_customer",
    "risk_prone_customer", "weekend_user", "night_driver",
]


def generate_personas(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Assign each row a soft latent persona derived from existing signal
    (account age, trust, transaction amount/risk) plus a random component,
    so persona assignment is probabilistic rather than a hard rule.
    """
    n = len(df)

    age_z = (df["account_age_days"] - df["account_age_days"].mean()) / df["account_age_days"].std()
    trust_z = (df["previous_trust_score"] - df["previous_trust_score"].mean()) / df["previous_trust_score"].std()
    amount_z = (df["transaction_amount"] - df["transaction_amount"].mean()) / df["transaction_amount"].std()
    risk_z = (df["transaction_risk"] - df["transaction_risk"].mean()) / df["transaction_risk"].std()
    noise = rng.normal(0, 1.0, size=n)

    composite = 0.35 * age_z + 0.30 * trust_z + 0.20 * amount_z - 0.15 * risk_z + 0.5 * noise

    # Bucket the composite score into persona-like quantile bands, with a
    # random draw resolving ties/edge rows into behavioral personas.
    bands = pd.qcut(composite.rank(method="first"), q=7, labels=False)
    random_persona = rng.choice(len(PERSONA_NAMES), size=n)

    # Blend: 70% driven by the composite band mapped to a persona group,
    # 30% purely random persona, to keep the assignment probabilistic.
    band_to_persona = {
        0: "new_customer", 1: "risk_prone_customer", 2: "weekend_user",
        3: "night_driver", 4: "frequent_traveler", 5: "corporate_employee",
        6: "trusted_commuter",
    }
    mapped = bands.map(band_to_persona)
    use_random = rng.random(n) < 0.30
    personas = mapped.where(~use_random, pd.Series(np.array(PERSONA_NAMES)[random_persona], index=df.index))

    # A slice of the wealthiest, most trusted band becomes high_value / senior
    high_value_mask = (amount_z > 1.0) & (trust_z > 0.5) & (rng.random(n) < 0.5)
    personas = personas.mask(high_value_mask, "high_value_customer")
    senior_mask = (age_z > 1.5) & (rng.random(n) < 0.4)
    personas = personas.mask(senior_mask, "senior_citizen")

    return personas.astype("category")


def apply_persona_effects(df: pd.DataFrame, personas: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """
    Let each persona subtly nudge a handful of features together, so a
    row's behavior reads as internally consistent rather than as
    independently-sampled columns. All effects are small multiplicative
    or additive perturbations, applied probabilistically per persona.
    """
    n = len(df)

    def nudge(col, mask, delta, noise_std=0.0):
        if mask.sum() == 0:
            return
        adj = delta + rng.normal(0, noise_std, size=mask.sum()) if noise_std else delta
        df.loc[mask, col] = _clip(df.loc[mask, col] + adj, col)

    m = personas == "trusted_commuter"
    nudge("time_familiarity", m, 0.03, 0.01)
    nudge("location_familiarity", m, 0.03, 0.01)
    nudge("stress_score", m, -0.02, 0.01)

    m = personas == "corporate_employee"
    nudge("command_familiarity", m, 0.02, 0.01)
    nudge("hesitation_score", m, -0.02, 0.01)

    m = personas == "frequent_traveler"
    nudge("location_familiarity", m, -0.05, 0.02)
    nudge("vehicle_speed", m, 2.0, 3.0)

    m = personas == "new_customer"
    nudge("previous_trust_score", m, -0.06, 0.02)
    nudge("hesitation_score", m, 0.04, 0.02)
    nudge("command_familiarity", m, -0.03, 0.02)

    m = personas == "senior_citizen"
    nudge("vehicle_speed", m, -4.0, 2.0)
    nudge("speech_rate_similarity", m, -0.02, 0.01)

    m = personas == "high_value_customer"
    nudge("transaction_amount", m, 0.0, 0.0)  # amount already reflects this persona
    nudge("successful_transactions", m, 60.0, 20.0)

    m = personas == "risk_prone_customer"
    nudge("transaction_risk", m, 0.03, 0.015)
    nudge("stress_score", m, 0.03, 0.015)

    m = personas == "weekend_user"
    nudge("time_familiarity", m, -0.02, 0.01)

    m = personas == "night_driver"
    nudge("vehicle_speed", m, 3.0, 4.0)
    nudge("location_familiarity", m, -0.02, 0.01)

    return df


# ============================================================================
# DISTRIBUTION REFINEMENT (skew / kurtosis / outlier repair)
# ============================================================================

def reshape_distribution(series: pd.Series, sampler: str, kwargs: dict,
                          low: float, high: float, rng: np.random.Generator) -> pd.Series:
    """
    Replace a column's marginal distribution with a healthier one drawn
    from `sampler`, while preserving the column's exact rank order (a
    strictly monotonic transform), so every relationship the column has
    with the rest of the dataset survives the reshape.
    """
    n = len(series)
    target = draw_sampler(sampler, kwargs, n, rng)
    target.sort()
    tmin, tmax = target.min(), target.max()
    if tmax > tmin:
        target = low + (target - tmin) / (tmax - tmin) * (high - low)
    else:
        target = np.full(n, (low + high) / 2.0)

    order = series.rank(method="first").to_numpy(dtype=int) - 1
    new_vals = target[order]
    return pd.Series(new_vals, index=series.index)


def refine_distributions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply order-preserving reshapes to every column flagged in Dataset_Info.md."""
    for col, spec in Config.RESHAPE_TARGETS.items():
        df[col] = reshape_distribution(
            df[col], spec["sampler"], spec["kwargs"], spec["low"], spec["high"], rng
        )
    return df


# ============================================================================
# DEPENDENCY REFINEMENT (rank-copula nudges)
# ============================================================================

def strengthen_dependency(df: pd.DataFrame, target: str, driver: str,
                           strength: float, direction: int, rng: np.random.Generator) -> pd.Series:
    """
    Blend `target`'s percentile rank a small amount toward (or away from,
    if direction=-1) `driver`'s percentile rank, then remap through
    `target`'s own empirical quantile function. This tightens the
    correlation between the two columns without ever producing a value
    outside the column's real observed range, and never makes the
    relationship deterministic (strength is always well under 1.0).
    """
    y_rank = _rank_pct(df[target], rng)
    x_rank = _rank_pct(df[driver], rng)
    if direction < 0:
        x_rank = 1.0 - x_rank

    blended = np.clip(y_rank * (1 - strength) + x_rank * strength, 0.001, 0.999)
    new_vals = df[target].quantile(blended).to_numpy()
    return pd.Series(new_vals, index=df.index)


def apply_dependencies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    for dep in Config.DEPENDENCIES:
        df[dep["target"]] = strengthen_dependency(
            df, dep["target"], dep["driver"], dep["strength"], dep["direction"], rng
        )
        df[dep["target"]] = _clip(df[dep["target"]], dep["target"])

    # fraud history -> transaction risk (categorical driver, not rank-based)
    boost = np.where(df["fraud_history"] > 0, rng.uniform(0.15, 0.35, size=len(df)), 0.0)
    df["transaction_risk"] = _clip(df["transaction_risk"] * (1 + boost), "transaction_risk")

    # failed_attempts -> transaction risk (more failed attempts, more risk)
    fa_effect = np.clip(df["failed_attempts"], 0, 5) * rng.uniform(0.01, 0.03, size=len(df))
    df["transaction_risk"] = _clip(df["transaction_risk"] + fa_effect, "transaction_risk")

    return df


# ============================================================================
# VEHICLE CONTEXT CONSISTENCY
# ============================================================================

def refine_vehicle_context(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """engine off -> speed ~ 0; driver absent -> seatbelt usually unfastened."""
    n = len(df)

    engine_off = df["engine_running"] == 0
    sensor_noise = np.abs(rng.normal(0, 0.4, size=n))
    df.loc[engine_off, "vehicle_speed"] = sensor_noise[engine_off.to_numpy()]

    driver_absent = df["driver_present"] == 0
    flip_prob = rng.random(n) < 0.85  # usually unfastened, not always
    unfasten_mask = driver_absent & flip_prob & (df["seatbelt_fastened"] == 1)
    df.loc[unfasten_mask, "seatbelt_fastened"] = 0.0

    return df


# ============================================================================
# BINARY IMBALANCE REFINEMENT
# ============================================================================

def refine_binary_balance(df: pd.DataFrame, personas: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """
    Nudge severely imbalanced binary flags with small, conditionally
    targeted flips so the minority class becomes slightly richer and more
    behaviorally grounded, without erasing the natural real-world
    imbalance (fraud, for instance, should stay rare).
    """
    n = len(df)

    # phone_verified: new customers are more likely to be unverified
    new_cust = personas == "new_customer"
    candidates = (df["phone_verified"] == 1) & new_cust
    max_flip = Config.BINARY_REFINEMENTS["phone_verified"]
    flip = candidates & (rng.random(n) < max_flip / max(new_cust.mean(), 1e-6))
    df.loc[flip, "phone_verified"] = 0.0

    # driver_present: slight increase in absent-driver cases (remote auth)
    candidates = df["driver_present"] == 1
    max_flip = Config.BINARY_REFINEMENTS["driver_present"]
    flip = candidates & (rng.random(n) < max_flip)
    df.loc[flip, "driver_present"] = 0.0

    # fraud_history: risk-prone persona slightly more likely to have history
    risk_prone = personas == "risk_prone_customer"
    candidates = (df["fraud_history"] == 0) & risk_prone & (df["transaction_risk"] > df["transaction_risk"].median())
    max_flip = Config.BINARY_REFINEMENTS["fraud_history"]
    flip = candidates & (rng.random(n) < max_flip / max(risk_prone.mean(), 1e-6))
    df.loc[flip, "fraud_history"] = 1.0

    # seatbelt_fastened: small general realism flip for high-stress rows
    candidates = (df["seatbelt_fastened"] == 1) & (df["stress_score"] > 0.5)
    max_flip = Config.BINARY_REFINEMENTS["seatbelt_fastened"]
    flip = candidates & (rng.random(n) < max_flip)
    df.loc[flip, "seatbelt_fastened"] = 0.0

    return df


# ============================================================================
# EDGE CASE INJECTION
# ============================================================================

def inject_edge_cases(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sprinkle a small number of realistic-but-uncommon combinations into
    the dataset so the model sees plausible tail scenarios, not just
    average behavior.
    """
    n = len(df)

    # 1) Trusted user under stress
    idx = rng.random(n) < Config.EDGE_CASE_FRACTION
    pool = df.index[(df["previous_trust_score"] > 0.75) & idx]
    if len(pool):
        df.loc[pool, "stress_score"] = _clip(
            df.loc[pool, "stress_score"] + rng.uniform(0.2, 0.35, size=len(pool)), "stress_score"
        )
        df.loc[pool, "hesitation_score"] = _clip(
            df.loc[pool, "hesitation_score"] + rng.uniform(0.1, 0.2, size=len(pool)), "hesitation_score"
        )

    # 2) Experienced customer in an unfamiliar location
    idx = rng.random(n) < Config.EDGE_CASE_FRACTION
    pool = df.index[(df["account_age_days"] > 2000) & idx]
    if len(pool):
        df.loc[pool, "location_familiarity"] = _clip(
            df.loc[pool, "location_familiarity"] - rng.uniform(0.3, 0.45, size=len(pool)), "location_familiarity"
        )

    # 3) High-value transaction with excellent authentication
    idx = rng.random(n) < Config.EDGE_CASE_FRACTION
    pool = df.index[(df["transaction_amount"] > df["transaction_amount"].quantile(0.85)) & idx]
    if len(pool):
        df.loc[pool, "speaker_similarity"] = _clip(
            df.loc[pool, "speaker_similarity"] + rng.uniform(0.005, 0.02, size=len(pool)), "speaker_similarity"
        )
        df.loc[pool, "liveness_score"] = _clip(
            df.loc[pool, "liveness_score"] + rng.uniform(0.01, 0.03, size=len(pool)), "liveness_score"
        )
        df.loc[pool, "spoof_probability"] = _clip(
            df.loc[pool, "spoof_probability"] * rng.uniform(0.5, 0.8, size=len(pool)), "spoof_probability"
        )

    # 4) Good speaker match despite a noisy cabin
    idx = rng.random(n) < Config.EDGE_CASE_FRACTION
    pool = df.index[(df["speaker_similarity"] > 0.97) & idx]
    if len(pool):
        df.loc[pool, "audio_quality"] = _clip(
            df.loc[pool, "audio_quality"] - rng.uniform(0.15, 0.3, size=len(pool)), "audio_quality"
        )

    return df


# ============================================================================
# NOISE INJECTION
# ============================================================================

def inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Small gaussian jitter on continuous [0,1]-style scores to avoid
    over-clean, obviously-synthetic value patterns."""
    n = len(df)
    for col in Config.JITTER_COLUMNS:
        std = df[col].std()
        if std == 0 or np.isnan(std):
            continue
        jitter = rng.normal(0, std * Config.JITTER_FRACTION_OF_STD, size=n)
        df[col] = _clip(df[col] + jitter, col)
    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate(original: pd.DataFrame, modified: pd.DataFrame) -> None:
    assert len(original) == len(modified), "Row count changed."
    assert list(original.columns) == list(modified.columns), "Column set/order changed."
    assert original.shape[1] == modified.shape[1], "Column count changed."
    assert modified.isna().sum().sum() == 0, "NaNs present after refinement."
    assert not modified.duplicated().any(), "Duplicate rows introduced."

    for col, (lo, hi) in Config.BOUNDS.items():
        col_min, col_max = modified[col].min(), modified[col].max()
        assert col_min >= lo - 1e-6, f"{col} below lower bound: {col_min} < {lo}"
        assert col_max <= hi + 1e-6, f"{col} above upper bound: {col_max} > {hi}"

    # no impossible combinations
    assert (modified.loc[modified["engine_running"] == 0, "vehicle_speed"] < 5.0).all(), \
        "Engine off but non-trivial vehicle speed detected."

    print("Validation passed: shape, dtypes, ranges, NaNs, duplicates, and logical consistency all OK.")


def restore_dtypes(modified: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Cast every column back to its original dtype (drop-in replacement)."""
    for col in original.columns:
        orig_dtype = original[col].dtype
        if pd.api.types.is_integer_dtype(orig_dtype):
            modified[col] = modified[col].round().astype(orig_dtype)
        elif pd.api.types.is_float_dtype(orig_dtype):
            modified[col] = modified[col].astype(orig_dtype)
        else:
            modified[col] = modified[col].astype(orig_dtype)
    return modified


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    rng = np.random.default_rng(Config.RANDOM_SEED)

    if not Config.INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Expected input dataset at {Config.INPUT_PATH.resolve()}. "
            "Run generate_transactions.py first."
        )

    original = pd.read_csv(Config.INPUT_PATH)
    assert list(original.columns) == Config.SCHEMA, (
        "Input schema does not match the declared schema; refusing to modify."
    )

    df = original.copy(deep=True)

    # Work in float64 for the whole transformation phase. Several steps
    # below (persona nudges, dependency/edge-case adjustments, jitter) add
    # small non-integer deltas to columns that are declared as int64 in the
    # original schema (successful_transactions, failed_attempts, ...).
    # pandas raises a `LossySetitemError` on an in-place float write into an
    # int64 column rather than silently truncating, so every numeric column
    # is upcast here; `restore_dtypes()` casts everything back to its exact
    # original dtype (rounding ints) once every transform has run.
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = df[col].astype("float64")

    # 1) Latent persona modeling (never persisted as columns)
    personas = generate_personas(df, rng)
    df = apply_persona_effects(df, personas, rng)

    # 2) Repair unhealthy marginal distributions (skew/kurtosis/outliers)
    df = refine_distributions(df, rng)

    # 3) Strengthen realistic dependencies between features
    df = apply_dependencies(df, rng)

    # 4) Vehicle-context logical consistency
    df = refine_vehicle_context(df, rng)

    # 5) Binary imbalance refinement
    df = refine_binary_balance(df, personas, rng)

    # 6) Sprinkle realistic edge cases
    df = inject_edge_cases(df, rng)

    # 7) Light noise to avoid overly clean synthetic patterns
    df = inject_noise(df, rng)

    # 8) Restore exact dtypes and column order, then validate
    df = df[Config.SCHEMA]
    df = restore_dtypes(df, original)
    validate(original, df)

    Config.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Config.OUTPUT_PATH, index=False)

    # ---- summary --------------------------------------------------------
    print("\nRefinement summary")
    print("=" * 60)
    print(f"Rows processed:              {len(df):,}")
    print(f"Columns (unchanged):         {len(df.columns)}")
    print(f"Distributions reshaped:      {list(Config.RESHAPE_TARGETS)}")
    print(f"Dependencies strengthened:   {len(Config.DEPENDENCIES) + 2}")
    print(f"Binary flags rebalanced:     {list(Config.BINARY_REFINEMENTS)}")
    print(f"Edge-case patterns injected: 4 (~{Config.EDGE_CASE_FRACTION*100:.1f}% each)")
    print(f"Output written to:           {Config.OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()