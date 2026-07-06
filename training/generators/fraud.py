"""
fraud.py
========

Fraud Scenario Framework for synthetic dataset generation.

Problem this solves
--------------------
Earlier versions of the generators (voice/behavior/vehicle/history/
transaction/intent) each accepted a single `fraudulent: bool` flag with a
hard-coded pair of distributions ("genuine" vs. "fraud"). That flag was
never actually wired up by `generate_users.py` / `generate_transactions.py`,
so every sample in the dataset was generated as genuine and the fraud
branches were dead code (see training/data/Dataset_Info.md -- spoof
probability never exceeded ~0.1 across 100k rows).

Even if it *had* been wired up, a single boolean is too coarse: real
attacks don't make every signal abnormal. A voice-replay attack degrades
voice biometrics but leaves vehicle telemetry and transaction behavior
almost untouched. A GPS-spoofing attack does the opposite. A boolean
can't express that; a scenario-aware, continuous-intensity model can.

Design
------
* `FraudScenario` -- an extensible enum of named attack types.
* `FraudTier` -- the "how sophisticated" bucket a scenario belongs to
  (opportunistic / sophisticated / coordinated), used only for sampling
  weights and reporting.
* `ScenarioSpec` -- static, declarative data describing one scenario: which
  subsystems it touches and how strongly (as a `(min, max)` impact range),
  plus optional per-feature multipliers for extra nuance within a
  subsystem. Adding a new scenario means adding one entry to `SCENARIOS`;
  no generator code changes.
* `FraudContext` -- the lightweight, per-sample object that actually flows
  through the pipeline (User -> Voice -> Behavior -> Vehicle -> History ->
  Transaction -> Intent). It exposes `feature_impact(subsystem, feature)`,
  a single float in [0, 1], which every generator uses to linearly blend
  its "genuine" and "fraud" distribution parameters. impact=0 reproduces
  the old genuine path exactly; impact=1 reproduces the old fraud path
  exactly; anything in between is a realistic, partially-degraded signal.
* `FraudGeneratorConfig` / `FraudScenarioSampler` -- central configuration
  (fraud rate, tier weights, severity range, correlation strength, seed)
  and the sampler that turns that configuration into a stream of
  `FraudContext` instances.

Ground-truth propagation
-------------------------
`FraudContext` is deliberately NOT merged into the model-facing feature
schema (users.csv / transactions.csv / dataset.csv keep their exact
existing columns). Real fraud detectors don't get told the attack type in
advance -- they have to infer it from evidence. Instead, `generate_users.py`
persists one row per user to a side-channel file (`fraud_context.csv`)
purely for (a) re-hydrating the same context when `generate_transactions.py`
builds the transaction/intent features for that user, and (b) offline
dataset validation/reporting (`analyze_fraud.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# ==========================================================
# Subsystems
# ==========================================================
# Every generator downstream of `generate_users.py` corresponds to exactly
# one of these. Keeping this as a single tuple (rather than re-typing
# strings everywhere) avoids typos silently creating a no-op impact key.

VOICE = "voice"
BEHAVIOR = "behavior"
VEHICLE = "vehicle"
HISTORY = "history"
TRANSACTION = "transaction"
INTENT = "intent"

SUBSYSTEMS: Tuple[str, ...] = (
    VOICE,
    BEHAVIOR,
    VEHICLE,
    HISTORY,
    TRANSACTION,
    INTENT,
)


# ==========================================================
# Enums
# ==========================================================

class FraudTier(str, Enum):
    """How sophisticated / coordinated an attack is."""

    GENUINE = "genuine"
    OPPORTUNISTIC = "opportunistic"
    SOPHISTICATED = "sophisticated"
    COORDINATED = "coordinated"


class FraudScenario(str, Enum):
    """
    Extensible catalogue of fraud scenarios.

    Adding a new scenario:
        1. Add a member here.
        2. Add a matching `ScenarioSpec` to `SCENARIOS` below.
    No generator code needs to change.
    """

    GENUINE = "genuine"

    # -- Opportunistic (low-effort, single-signal attacks) -----------
    VOICE_REPLAY = "voice_replay_attack"
    GPS_SPOOFING = "gps_spoofing"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"

    # -- Sophisticated (crafted, multi-signal attacks) ----------------
    DEEPFAKE_VOICE = "deepfake_voice_attack"
    STOLEN_DEVICE = "stolen_device"
    ACCOUNT_TAKEOVER = "account_takeover"

    # -- Coordinated (highest effort, broad-signal attacks) -----------
    INSIDER_FRAUD = "insider_fraud"
    MULTI_STAGE_COORDINATED = "multi_stage_coordinated_attack"


# ==========================================================
# Scenario specifications (declarative, extensible table)
# ==========================================================

ImpactRange = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Static definition of how one scenario affects the pipeline."""

    scenario: FraudScenario
    tier: FraudTier
    weight: float  # relative sampling weight within its tier
    subsystem_impact: Mapping[str, ImpactRange]
    feature_multipliers: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    description: str = ""

    def impact_range(self, subsystem: str) -> ImpactRange:
        return self.subsystem_impact.get(subsystem, (0.0, 0.0))


# Every scenario declares, per subsystem, the *range* of impact it can
# produce (sampled per-instance for variability) and, optionally, per
# feature multipliers so two scenarios that both touch "voice" can still
# look different (e.g. deepfake keeps audio_quality clean; replay does not).
SCENARIOS: Dict[FraudScenario, ScenarioSpec] = {
    FraudScenario.GENUINE: ScenarioSpec(
        scenario=FraudScenario.GENUINE,
        tier=FraudTier.GENUINE,
        weight=0.0,
        subsystem_impact={s: (0.0, 0.0) for s in SUBSYSTEMS},
        description="No attack. All subsystems generate genuine behavior.",
    ),
    FraudScenario.VOICE_REPLAY: ScenarioSpec(
        scenario=FraudScenario.VOICE_REPLAY,
        tier=FraudTier.OPPORTUNISTIC,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.55, 1.0),
            BEHAVIOR: (0.0, 0.15),
            VEHICLE: (0.0, 0.05),
            HISTORY: (0.0, 0.10),
            TRANSACTION: (0.05, 0.25),
            INTENT: (0.0, 0.15),
        },
        feature_multipliers={
            VOICE: {
                "audio_quality": 0.9,      # recording artifacts degrade audio
                "liveness_score": 1.0,     # replay fails liveness hard
                "speaker_similarity": 0.8,
                "spoof_probability": 1.0,
            },
        },
        description="A pre-recorded genuine voice sample is replayed.",
    ),
    FraudScenario.GPS_SPOOFING: ScenarioSpec(
        scenario=FraudScenario.GPS_SPOOFING,
        tier=FraudTier.OPPORTUNISTIC,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.0, 0.05),
            BEHAVIOR: (0.05, 0.25),
            VEHICLE: (0.55, 1.0),
            HISTORY: (0.0, 0.05),
            TRANSACTION: (0.05, 0.20),
            INTENT: (0.0, 0.10),
        },
        feature_multipliers={
            VEHICLE: {
                "location_familiarity": 1.0,
                "time_familiarity": 0.8,
                # spoofing the GPS feed doesn't change the vehicle's real
                # physical sensors, so leave these near-genuine:
                "vehicle_speed": 0.15,
                "engine_running": 0.0,
                "driver_present": 0.0,
                "seatbelt_fastened": 0.0,
            },
        },
        description="Location/time telemetry is spoofed; physical vehicle "
        "sensors remain genuine.",
    ),
    FraudScenario.BEHAVIORAL_ANOMALY: ScenarioSpec(
        scenario=FraudScenario.BEHAVIORAL_ANOMALY,
        tier=FraudTier.OPPORTUNISTIC,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.0, 0.15),
            BEHAVIOR: (0.55, 1.0),
            VEHICLE: (0.05, 0.20),
            HISTORY: (0.0, 0.10),
            TRANSACTION: (0.0, 0.15),
            INTENT: (0.0, 0.10),
        },
        description="Genuine speaker under duress / coaching / unfamiliar "
        "command usage -- behavioral biometrics degrade, voice does not.",
    ),
    FraudScenario.DEEPFAKE_VOICE: ScenarioSpec(
        scenario=FraudScenario.DEEPFAKE_VOICE,
        tier=FraudTier.SOPHISTICATED,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.65, 1.0),
            BEHAVIOR: (0.0, 0.20),
            VEHICLE: (0.0, 0.05),
            HISTORY: (0.0, 0.10),
            TRANSACTION: (0.10, 0.35),
            INTENT: (0.0, 0.20),
        },
        feature_multipliers={
            VOICE: {
                "audio_quality": 0.3,      # synthetic audio is often clean
                "liveness_score": 1.0,     # but liveness/anti-spoof catches it
                "speaker_similarity": 0.55,
                "spoof_probability": 1.0,
            },
        },
        description="AI-generated/cloned voice; high audio quality but "
        "fails liveness and anti-spoofing checks.",
    ),
    FraudScenario.STOLEN_DEVICE: ScenarioSpec(
        scenario=FraudScenario.STOLEN_DEVICE,
        tier=FraudTier.SOPHISTICATED,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.15, 0.45),
            BEHAVIOR: (0.35, 0.75),
            VEHICLE: (0.25, 0.60),
            HISTORY: (0.05, 0.25),
            TRANSACTION: (0.15, 0.45),
            INTENT: (0.0, 0.20),
        },
        description="A different physical person operates a stolen, "
        "already-enrolled device in an unfamiliar context.",
    ),
    FraudScenario.ACCOUNT_TAKEOVER: ScenarioSpec(
        scenario=FraudScenario.ACCOUNT_TAKEOVER,
        tier=FraudTier.SOPHISTICATED,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.25, 0.65),
            BEHAVIOR: (0.25, 0.65),
            VEHICLE: (0.10, 0.40),
            HISTORY: (0.0, 0.20),
            TRANSACTION: (0.30, 0.70),
            INTENT: (0.10, 0.30),
        },
        description="Attacker gains control of a legitimate, verified "
        "account -- multiple moderately-abnormal signals at once.",
    ),
    FraudScenario.INSIDER_FRAUD: ScenarioSpec(
        scenario=FraudScenario.INSIDER_FRAUD,
        tier=FraudTier.COORDINATED,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.0, 0.10),
            BEHAVIOR: (0.0, 0.20),
            VEHICLE: (0.0, 0.10),
            HISTORY: (0.0, 0.15),
            TRANSACTION: (0.45, 0.85),
            INTENT: (0.0, 0.15),
        },
        description="The legitimate account holder authenticates normally; "
        "the abnormal signal is almost entirely in transaction behavior.",
    ),
    FraudScenario.MULTI_STAGE_COORDINATED: ScenarioSpec(
        scenario=FraudScenario.MULTI_STAGE_COORDINATED,
        tier=FraudTier.COORDINATED,
        weight=1.0,
        subsystem_impact={
            VOICE: (0.45, 0.90),
            BEHAVIOR: (0.40, 0.80),
            VEHICLE: (0.30, 0.70),
            HISTORY: (0.20, 0.50),
            TRANSACTION: (0.45, 0.90),
            INTENT: (0.20, 0.50),
        },
        description="Full-spectrum coordinated attack -- broad, strong "
        "abnormality across nearly every subsystem.",
    ),
}


FRAUD_SCENARIOS: Tuple[FraudScenario, ...] = tuple(
    s for s in FraudScenario if s is not FraudScenario.GENUINE
)


def scenarios_in_tier(tier: FraudTier) -> Tuple[FraudScenario, ...]:
    return tuple(
        spec.scenario for spec in SCENARIOS.values() if spec.tier is tier
    )


# ==========================================================
# Blending helper
# ==========================================================

def blend(genuine: float, fraud: float, impact: float) -> float:
    """Linearly interpolate a distribution parameter between its genuine
    and fraud values. impact=0 -> genuine, impact=1 -> fraud."""
    impact = float(np.clip(impact, 0.0, 1.0))
    return genuine + (fraud - genuine) * impact


# ==========================================================
# FraudContext -- the object threaded through the pipeline
# ==========================================================

@dataclass(slots=True)
class FraudContext:
    scenario: FraudScenario
    tier: FraudTier
    intensity: float
    subsystem_impact: Dict[str, float]
    feature_multipliers: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------

    @property
    def is_fraud(self) -> bool:
        return self.scenario is not FraudScenario.GENUINE

    def impact(self, subsystem: str) -> float:
        return self.subsystem_impact.get(subsystem, 0.0)

    def feature_impact(
        self, subsystem: str, feature: str, default_multiplier: float = 1.0
    ) -> float:
        """The effective [0, 1] impact a single generator feature should
        use, i.e. the subsystem-level impact scaled by an optional
        per-feature multiplier (for within-subsystem nuance)."""
        base = self.impact(subsystem)
        mult = self.feature_multipliers.get(subsystem, {}).get(
            feature, default_multiplier
        )
        return float(np.clip(base * mult, 0.0, 1.0))

    # ------------------------------------------------------------------

    @classmethod
    def genuine(cls) -> "FraudContext":
        return cls(
            scenario=FraudScenario.GENUINE,
            tier=FraudTier.GENUINE,
            intensity=0.0,
            subsystem_impact={s: 0.0 for s in SUBSYSTEMS},
            feature_multipliers={},
        )

    @classmethod
    def legacy(cls, fraudulent: bool) -> "FraudContext":
        """Backward-compat shim for the old `fraudulent: bool` API. Maps
        `False` -> genuine and `True` -> a full-impact, generic fraud
        context so every existing call site keeps working unchanged."""
        if not fraudulent:
            return cls.genuine()
        return cls(
            scenario=FraudScenario.ACCOUNT_TAKEOVER,
            tier=FraudTier.SOPHISTICATED,
            intensity=1.0,
            subsystem_impact={s: 1.0 for s in SUBSYSTEMS},
            feature_multipliers={},
        )

    @classmethod
    def resolve(
        cls,
        fraudulent: bool = False,
        fraud_context: Optional["FraudContext"] = None,
    ) -> "FraudContext":
        """Every generator calls this once to decide which context to use:
        an explicit `fraud_context` always wins; otherwise fall back to the
        legacy boolean (default: genuine)."""
        if fraud_context is not None:
            return fraud_context
        return cls.legacy(fraudulent)

    # ------------------------------------------------------------------
    # CSV side-channel serialization
    # ------------------------------------------------------------------

    def to_row(self, user_id: int) -> Dict[str, object]:
        row: Dict[str, object] = {
            "user_id": user_id,
            "scenario": self.scenario.value,
            "tier": self.tier.value,
            "intensity": round(float(self.intensity), 4),
        }
        for subsystem in SUBSYSTEMS:
            row[f"impact_{subsystem}"] = round(
                float(self.subsystem_impact.get(subsystem, 0.0)), 4
            )
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "FraudContext":
        scenario = FraudScenario(row["scenario"])
        tier = FraudTier(row["tier"])
        impacts = {
            s: float(row.get(f"impact_{s}", 0.0)) for s in SUBSYSTEMS
        }
        feature_multipliers = SCENARIOS[scenario].feature_multipliers
        return cls(
            scenario=scenario,
            tier=tier,
            intensity=float(row["intensity"]),
            subsystem_impact=impacts,
            feature_multipliers=feature_multipliers,
        )


FRAUD_CONTEXT_COLUMNS: Tuple[str, ...] = (
    "user_id",
    "scenario",
    "tier",
    "intensity",
    *(f"impact_{s}" for s in SUBSYSTEMS),
)


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class FraudGeneratorConfig:
    """
    Central, override-able configuration for the fraud simulation.

    Parameters
    ----------
    random_seed:
        Seed for the fraud sampler's own RNG stream (independent of the
        per-generator RNGs, so toggling fraud config doesn't reshuffle
        genuine feature draws).
    fraud_rate:
        Overall probability that a given user/transaction is fraudulent
        at all (default ~5.5%, matching Genuine ~94.5%).
    tier_weights:
        Relative weight of each non-genuine tier. Normalized internally.
        Defaults approximate: opportunistic ~2.7%, sophisticated ~1.9%,
        coordinated ~0.8% of the *total* population.
    severity_range:
        (min, max) overall intensity sampled per fraud instance. Scales
        every subsystem impact down/up together.
    correlation_strength:
        In [0, 1]. How strongly a scenario's per-subsystem impacts move
        together for a single instance (0 = fully independent per
        subsystem within their ranges, 1 = fully correlated / lock-step).
        Coordinated attacks look more realistic with impacts that co-move.
    scenario_weight_overrides:
        Optional explicit override of a scenario's sampling weight within
        its tier (default: use `ScenarioSpec.weight`).
    """

    random_seed: int = 42
    fraud_rate: float = 0.055
    tier_weights: Dict[FraudTier, float] = field(
        default_factory=lambda: {
            FraudTier.OPPORTUNISTIC: 0.50,
            FraudTier.SOPHISTICATED: 0.35,
            FraudTier.COORDINATED: 0.15,
        }
    )
    severity_range: ImpactRange = (0.55, 1.0)
    correlation_strength: float = 0.65
    scenario_weight_overrides: Dict[FraudScenario, float] = field(
        default_factory=dict
    )

    def scenario_weight(self, scenario: FraudScenario) -> float:
        return self.scenario_weight_overrides.get(
            scenario, SCENARIOS[scenario].weight
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "random_seed": self.random_seed,
            "fraud_rate": self.fraud_rate,
            "tier_weights": {k.value: v for k, v in self.tier_weights.items()},
            "severity_range": list(self.severity_range),
            "correlation_strength": self.correlation_strength,
            "scenario_weight_overrides": {
                k.value: v for k, v in self.scenario_weight_overrides.items()
            },
        }


# ==========================================================
# Sampler (Strategy: config-driven, no branching per scenario)
# ==========================================================

class FraudScenarioSampler:
    """Turns a `FraudGeneratorConfig` into a stream of `FraudContext`
    instances with the requested tier/scenario distribution."""

    def __init__(
        self,
        config: Optional[FraudGeneratorConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config or FraudGeneratorConfig()
        self.rng = rng or np.random.default_rng(self.config.random_seed)

        self._tiers: Sequence[FraudTier] = (
            FraudTier.OPPORTUNISTIC,
            FraudTier.SOPHISTICATED,
            FraudTier.COORDINATED,
        )
        self._tier_probs = self._normalize(
            [self.config.tier_weights.get(t, 0.0) for t in self._tiers]
        )
        self._tier_scenarios: Dict[FraudTier, Tuple[FraudScenario, ...]] = {
            t: scenarios_in_tier(t) for t in self._tiers
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(weights: Iterable[float]) -> np.ndarray:
        arr = np.array(list(weights), dtype=float)
        total = arr.sum()
        if total <= 0:
            return np.full(len(arr), 1.0 / max(len(arr), 1))
        return arr / total

    def _sample_tier(self) -> FraudTier:
        idx = self.rng.choice(len(self._tiers), p=self._tier_probs)
        return self._tiers[idx]

    def _sample_scenario(self, tier: FraudTier) -> FraudScenario:
        candidates = self._tier_scenarios[tier]
        weights = self._normalize(
            [self.config.scenario_weight(s) for s in candidates]
        )
        idx = self.rng.choice(len(candidates), p=weights)
        return candidates[idx]

    def _sample_subsystem_impact(
        self, spec: ScenarioSpec, intensity: float
    ) -> Dict[str, float]:
        # A single shared standard-normal draw per instance lets multiple
        # subsystems co-move (higher correlation_strength => more
        # lock-step), which is what makes a "coordinated" attack look
        # coordinated rather than a grab-bag of independent randomness.
        shared = self.rng.normal(0.0, 1.0)
        shared_unit = 1.0 / (1.0 + np.exp(-shared))  # sigmoid -> (0, 1)

        impacts: Dict[str, float] = {}
        for subsystem in SUBSYSTEMS:
            lo, hi = spec.impact_range(subsystem)
            if hi <= lo:
                impacts[subsystem] = 0.0
                continue
            independent_draw = self.rng.uniform(lo, hi)
            correlated_draw = lo + (hi - lo) * shared_unit
            blended = (
                self.config.correlation_strength * correlated_draw
                + (1.0 - self.config.correlation_strength) * independent_draw
            )
            impacts[subsystem] = float(np.clip(blended * intensity, 0.0, 1.0))
        return impacts

    # ------------------------------------------------------------------

    def sample(self) -> FraudContext:
        if self.rng.random() >= self.config.fraud_rate:
            return FraudContext.genuine()

        tier = self._sample_tier()
        scenario = self._sample_scenario(tier)
        spec = SCENARIOS[scenario]
        intensity = float(self.rng.uniform(*self.config.severity_range))
        subsystem_impact = self._sample_subsystem_impact(spec, intensity)

        return FraudContext(
            scenario=scenario,
            tier=tier,
            intensity=intensity,
            subsystem_impact=subsystem_impact,
            feature_multipliers=spec.feature_multipliers,
        )

    def sample_many(self, n: int) -> list[FraudContext]:
        return [self.sample() for _ in range(n)]
