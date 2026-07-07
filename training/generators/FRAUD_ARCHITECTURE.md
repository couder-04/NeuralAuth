# Fraud Simulation Framework

This document describes the fraud-scenario architecture used by the
synthetic dataset generation pipeline
(`generate_users.py` -> `generate_transactions.py` -> `modify_dataset.py`
-> `generate_labels.py`), implemented in `generators/fraud.py`.

## Why this exists

The low-level generators (`voice.py`, `behavior.py`, `vehicle.py`,
`history.py`, `transaction.py`, `intent.py`) always had a `fraudulent:
bool` parameter with hard-coded "genuine" and "fraud" distributions. It
was never actually wired up by `generate_users.py` /
`generate_transactions.py`, so every row in the dataset was generated as
genuine. The label generator (`generate_labels.py`) still produced a
`Fraudster` persona and REJECT decisions, but only from rare tail
combinations of otherwise-legitimate feature ranges and a static
`fraud_history` flag -- never from real degraded biometric/behavioral
evidence. `training/data/Dataset_Info.md` confirms this empirically:
`spoof_probability` never exceeded ~0.1 and `liveness_score` never
dropped below ~0.84 across 100k generated rows.

This framework replaces the unused boolean with a scenario-aware,
continuous-intensity fraud simulation that actually reaches the raw
features.

## Core concepts

### `FraudScenario` (enum)

An extensible catalogue of named attack types, grouped into `FraudTier`s
by sophistication:

| Tier | Scenarios |
|---|---|
| Opportunistic | `voice_replay_attack`, `gps_spoofing`, `behavioral_anomaly` |
| Sophisticated | `deepfake_voice_attack`, `stolen_device`, `account_takeover` |
| Coordinated | `insider_fraud`, `multi_stage_coordinated_attack` |

Adding a new scenario: add a member to `FraudScenario`, then add a
matching `ScenarioSpec` to the `SCENARIOS` table. No generator code needs
to change.

### `ScenarioSpec` (declarative table, not branching logic)

Each scenario declares, per subsystem (`voice` / `behavior` / `vehicle` /
`history` / `transaction` / `intent`), a `(min, max)` **impact range** --
how strongly an instance of that scenario can push that subsystem away
from genuine. A subsystem with `(0.0, 0.0)` is left completely untouched.
Optional `feature_multipliers` add nuance *within* a subsystem (e.g. a
deepfake keeps `audio_quality` clean while still failing `liveness_score`
hard; a replay attack degrades both).

This is what produces *correlated evidence* rather than "every feature is
abnormal": a `gps_spoofing` instance has near-zero voice/history impact
and high vehicle impact; `insider_fraud` is the mirror image (transaction
behavior only); `multi_stage_coordinated_attack` is broad and strong
across everything.

### `FraudContext` (threaded through the pipeline)

The per-sample object that actually flows: `User -> Voice -> Behavior ->
Vehicle -> History -> Transaction -> Intent`. It resolves a scenario's
declared ranges into concrete per-instance numbers (scenario, tier,
overall `intensity`, and a resolved `subsystem_impact` dict) and exposes:

```python
ctx.feature_impact(subsystem, feature)  # -> float in [0, 1]
```

Every generator uses this single float to linearly blend its genuine and
fraud distribution parameters:

```python
impact = ctx.feature_impact("voice", "liveness_score")
mean = blend(0.97, 0.35, impact)   # genuine mean -> fraud mean
std  = blend(0.03, 0.18, impact)
liveness_score = clip(rng.normal(mean, std), 0, 1)
```

`impact = 0.0` reproduces the old "genuine" branch exactly. `impact =
1.0` reproduces the old "fraud" branch exactly. Anything in between is a
realistic, partially-degraded signal -- there is no giant if/else per
generator anymore, just one blend per feature.

### Backward compatibility

Every generator's public function still accepts the original `fraudulent:
bool = False` parameter and behaves exactly as before if `fraud_context`
is omitted (`FraudContext.legacy(bool)` maps `False` -> genuine and `True`
-> a full-impact generic-fraud context). Existing call sites, docs, and
scripts do not need to change.

### `FraudGeneratorConfig` / `FraudScenarioSampler`

Central, override-able configuration:

```python
@dataclass
class FraudGeneratorConfig:
    random_seed: int = 42
    fraud_rate: float = 0.055                 # ~94.5% genuine
    tier_weights: dict = {...}                 # opportunistic/sophisticated/coordinated split
    severity_range: tuple = (0.55, 1.0)        # per-instance overall intensity
    correlation_strength: float = 0.65         # how tightly subsystem impacts co-move
    scenario_weight_overrides: dict = {}       # per-scenario sampling weight override
```

`FraudScenarioSampler(config).sample()` returns one `FraudContext`,
weighted first by `fraud_rate` (genuine vs. any fraud), then by
`tier_weights`, then by each scenario's relative `weight` within its
tier. `correlation_strength` controls how strongly a single instance's
per-subsystem impacts move together (via a shared, per-instance draw) --
higher values make "coordinated" attacks look coordinated rather than a
grab-bag of independently-rolled subsystem impacts.

Default realized population (verified empirically at n=200k in
`tests/test_fraud_generation.py`): **Genuine ~94.5%, Opportunistic ~2.7%,
Sophisticated ~1.9%, Coordinated ~0.8%** -- matching the brief's target
ranges.

## Propagation without changing existing schemas

`generate_users.py` samples one `FraudContext` per user and threads it
into `generate_voice` / `generate_behavior` / `generate_vehicle` /
`generate_history`. `generate_identity` is deliberately **not**
fraud-coupled: an attacker who steals a device or takes over an account
does not change that account's real KYC/verification history.

The context is written to a new side-channel file,
`training/data/fraud_context.csv` (`user_id, scenario, tier, intensity,
impact_voice, impact_behavior, impact_vehicle, impact_history,
impact_transaction, impact_intent`) -- **not** merged into
`users.csv`/`transactions.csv`/`dataset.csv`. Two reasons:

1. **No label leakage.** A real fraud detector has to infer the attack
   from evidence; it should never see "this row's scenario is
   `gps_spoofing`" as an input feature.
2. **Zero schema churn.** Every existing column-order assertion in
   `generate_transactions.py` / `modify_dataset.py` continues to pass
   unmodified.

`generate_transactions.py` re-hydrates the same `FraudContext` per
`user_id` from `fraud_context.csv` and threads it into
`generate_transaction` / `generate_intent`, so one user's evidence is
consistent across every subsystem rather than being re-rolled
independently at each pipeline stage. If `fraud_context.csv` is missing
(e.g. an older `users.csv`), every user falls back to genuine rather than
failing the run.

## `generate_labels.py` needed no changes

The label engine already derives `trust_score` / `risk_score` /
`decision` purely from biometric/behavioral/transaction features -- never
from a fraud flag. Once those features actually carry real fraud
evidence, the existing risk engine (and its iterative logit calibration
that holds the ALLOW/CHALLENGE/OTP/REJECT distribution to ~62/20/12/6%)
responds to it automatically: fraud scenarios show measurably higher
`risk_score` and a higher REJECT/CHALLENGE rate than genuine, without
"fraud always means REJECT" being hard-coded anywhere. See
`Fraud_Info.md`'s "Decision Breakdown Per Scenario" for the empirical
numbers on a generated sample.

## One bug this exposed in `modify_dataset.py`

`modify_dataset.py`'s `RESHAPE_TARGETS` used to force
`speaker_similarity` into `[0.9123, 1.0]` and `transaction_risk` into
`[0.0, 0.60]` via a rank-preserving remap. Those ranges were tuned for
the old fraud-less data (where both columns were unnaturally narrow) and
would have rank-mapped the newly-injected fraud tail right back into a
"healthy-looking" narrow band -- silently erasing the separation this
framework introduces. Both entries have been removed (the post-fraud
distributions no longer need a synthetic reshape); see the comment above
`RESHAPE_TARGETS` in that file.

## Validation

`training/analyze_fraud.py` reads `fraud_context.csv` + `dataset.csv` and
writes `training/data/Fraud_Info.md`, covering: per-scenario counts and a
"did every scenario actually fire" check, tier/fraud-rate balance vs.
configuration, per-subsystem feature comparisons (fraud vs. genuine),
spoof/liveness distribution tails, Pearson correlation of fraud presence
with `trust_score`/`risk_score`/`decision`, overall and fraud-conditioned
decision class balance, and a full per-scenario decision breakdown.

## Extending

To add a new scenario (e.g. "SIM swap"):

1. Add `SIM_SWAP = "sim_swap"` to `FraudScenario`.
2. Add a `ScenarioSpec` to `SCENARIOS` with its tier, subsystem impact
   ranges, and any feature multipliers.
3. Nothing else changes -- the sampler, the generators, the CSV
   round-trip, and `analyze_fraud.py` all pick it up automatically.

To change the overall fraud rate or tier mix, construct a
`FraudGeneratorConfig` with the desired values and pass it to
`generate_users.py --fraud-rate ... --seed ...`, or edit the defaults in
`FraudGeneratorConfig` directly.
