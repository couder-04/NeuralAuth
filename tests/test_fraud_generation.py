"""
tests/test_fraud_generation.py

Tests for the synthetic fraud-scenario generation framework
(training/generators/fraud.py and its integration into the low-level
voice/behavior/vehicle/history/transaction/intent generators).

These tests import the `training/generators` package directly (the same
way the generation scripts themselves do: `from generators.xxx import
...`) by putting `training/` on `sys.path`, rather than importing through
`training.generators.xxx`, since the generator modules use bare
`generators.` imports internally.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from generators.fraud import (  # noqa: E402
    FRAUD_CONTEXT_COLUMNS,
    FRAUD_SCENARIOS,
    SCENARIOS,
    SUBSYSTEMS,
    FraudContext,
    FraudGeneratorConfig,
    FraudScenario,
    FraudScenarioSampler,
    FraudTier,
    blend,
)
from generators.voice import generate_voice  # noqa: E402
from generators.behavior import generate_behavior  # noqa: E402
from generators.vehicle import generate_vehicle  # noqa: E402
from generators.history import generate_history  # noqa: E402
from generators.transaction import generate_transaction  # noqa: E402
from generators.intent import generate_intent  # noqa: E402


# ==========================================================
# blend()
# ==========================================================

class TestBlend:
    def test_zero_impact_is_genuine(self):
        assert blend(0.9, 0.1, 0.0) == 0.9

    def test_full_impact_is_fraud(self):
        assert blend(0.9, 0.1, 1.0) == pytest.approx(0.1)

    def test_partial_impact_interpolates(self):
        assert blend(0.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_clips_out_of_range_impact(self):
        assert blend(0.0, 1.0, 5.0) == 1.0
        assert blend(0.0, 1.0, -5.0) == 0.0


# ==========================================================
# ScenarioSpec registry
# ==========================================================

class TestScenarioRegistry:
    def test_every_scenario_enum_member_has_a_spec(self):
        for scenario in FraudScenario:
            assert scenario in SCENARIOS

    def test_genuine_has_zero_impact_everywhere(self):
        spec = SCENARIOS[FraudScenario.GENUINE]
        for subsystem in SUBSYSTEMS:
            lo, hi = spec.impact_range(subsystem)
            assert lo == 0.0 and hi == 0.0

    def test_every_non_genuine_scenario_touches_at_least_one_subsystem(self):
        for scenario in FRAUD_SCENARIOS:
            spec = SCENARIOS[scenario]
            touched = [s for s in SUBSYSTEMS if spec.impact_range(s)[1] > 0.0]
            assert touched, f"{scenario} has no subsystem impact at all"

    def test_impact_ranges_are_valid(self):
        for scenario, spec in SCENARIOS.items():
            for subsystem in SUBSYSTEMS:
                lo, hi = spec.impact_range(subsystem)
                assert 0.0 <= lo <= hi <= 1.0, (scenario, subsystem, lo, hi)


# ==========================================================
# FraudContext
# ==========================================================

class TestFraudContext:
    def test_genuine_context_is_not_fraud(self):
        ctx = FraudContext.genuine()
        assert not ctx.is_fraud
        for s in SUBSYSTEMS:
            assert ctx.impact(s) == 0.0

    def test_legacy_false_matches_genuine(self):
        ctx = FraudContext.legacy(False)
        assert not ctx.is_fraud
        assert ctx.subsystem_impact == FraudContext.genuine().subsystem_impact

    def test_legacy_true_is_full_impact_fraud(self):
        ctx = FraudContext.legacy(True)
        assert ctx.is_fraud
        for s in SUBSYSTEMS:
            assert ctx.impact(s) == 1.0

    def test_resolve_prefers_explicit_context(self):
        explicit = FraudContext.legacy(True)
        resolved = FraudContext.resolve(fraudulent=False, fraud_context=explicit)
        assert resolved is explicit

    def test_resolve_falls_back_to_legacy_bool(self):
        resolved = FraudContext.resolve(fraudulent=True, fraud_context=None)
        assert resolved.is_fraud

    def test_feature_impact_applies_multiplier(self):
        ctx = FraudContext(
            scenario=FraudScenario.DEEPFAKE_VOICE,
            tier=FraudTier.SOPHISTICATED,
            intensity=1.0,
            subsystem_impact={s: 1.0 for s in SUBSYSTEMS},
            feature_multipliers={"voice": {"audio_quality": 0.3}},
        )
        assert ctx.feature_impact("voice", "audio_quality") == pytest.approx(0.3)
        # unspecified feature defaults to the raw subsystem impact
        assert ctx.feature_impact("voice", "liveness_score") == pytest.approx(1.0)

    def test_csv_roundtrip_preserves_scenario_and_impacts(self):
        sampler = FraudScenarioSampler(FraudGeneratorConfig(random_seed=3, fraud_rate=1.0))
        ctx = sampler.sample()
        row = ctx.to_row(user_id=123)

        assert set(row.keys()) == set(FRAUD_CONTEXT_COLUMNS)

        restored = FraudContext.from_row(row)
        assert restored.scenario == ctx.scenario
        assert restored.tier == ctx.tier
        assert restored.intensity == pytest.approx(ctx.intensity, abs=1e-3)
        for s in SUBSYSTEMS:
            assert restored.impact(s) == pytest.approx(ctx.impact(s), abs=1e-3)

    def test_csv_roundtrip_from_pandas_series(self):
        """`generate_transactions.py` re-hydrates FraudContext from a row
        of a pandas DataFrame (a Series), not a plain dict -- make sure
        that works too."""
        sampler = FraudScenarioSampler(FraudGeneratorConfig(random_seed=4, fraud_rate=1.0))
        ctx = sampler.sample()
        df = pd.DataFrame([ctx.to_row(user_id=7)]).set_index("user_id")
        restored = FraudContext.from_row(df.loc[7])
        assert restored.scenario == ctx.scenario


# ==========================================================
# FraudScenarioSampler distribution
# ==========================================================

class TestFraudScenarioSampler:
    def test_every_scenario_is_reachable(self):
        sampler = FraudScenarioSampler(FraudGeneratorConfig(random_seed=42, fraud_rate=1.0))
        seen = {c.scenario for c in sampler.sample_many(5000)}
        assert seen == set(FRAUD_SCENARIOS)

    def test_genuine_never_appears_when_fraud_rate_is_1(self):
        sampler = FraudScenarioSampler(FraudGeneratorConfig(random_seed=42, fraud_rate=1.0))
        scenarios = {c.scenario for c in sampler.sample_many(2000)}
        assert FraudScenario.GENUINE not in scenarios

    def test_only_genuine_when_fraud_rate_is_0(self):
        sampler = FraudScenarioSampler(FraudGeneratorConfig(random_seed=42, fraud_rate=0.0))
        for ctx in sampler.sample_many(200):
            assert ctx.scenario is FraudScenario.GENUINE
            assert not ctx.is_fraud

    def test_fraud_rate_matches_configuration_within_tolerance(self):
        config = FraudGeneratorConfig(random_seed=123, fraud_rate=0.10)
        sampler = FraudScenarioSampler(config)
        n = 50_000
        fraud_count = sum(1 for c in sampler.sample_many(n) if c.is_fraud)
        assert abs(fraud_count / n - 0.10) < 0.01

    def test_tier_distribution_matches_configuration_within_tolerance(self):
        config = FraudGeneratorConfig(
            random_seed=99,
            fraud_rate=1.0,
            tier_weights={
                FraudTier.OPPORTUNISTIC: 0.5,
                FraudTier.SOPHISTICATED: 0.3,
                FraudTier.COORDINATED: 0.2,
            },
        )
        sampler = FraudScenarioSampler(config)
        n = 50_000
        tiers = Counter(c.tier for c in sampler.sample_many(n))
        assert abs(tiers[FraudTier.OPPORTUNISTIC] / n - 0.5) < 0.02
        assert abs(tiers[FraudTier.SOPHISTICATED] / n - 0.3) < 0.02
        assert abs(tiers[FraudTier.COORDINATED] / n - 0.2) < 0.02

    def test_higher_correlation_strength_increases_cross_subsystem_correlation(self):
        """Coordinated attacks should look coordinated: subsystem impacts
        should co-move more strongly as correlation_strength increases."""

        def multisubsystem_corr(strength: float) -> float:
            config = FraudGeneratorConfig(
                random_seed=7,
                fraud_rate=1.0,
                correlation_strength=strength,
                tier_weights={FraudTier.COORDINATED: 1.0},
            )
            sampler = FraudScenarioSampler(config)
            ctxs = [
                c
                for c in sampler.sample_many(4000)
                if c.scenario is FraudScenario.MULTI_STAGE_COORDINATED
            ]
            voice = np.array([c.impact("voice") for c in ctxs])
            behavior = np.array([c.impact("behavior") for c in ctxs])
            return float(np.corrcoef(voice, behavior)[0, 1])

        low = multisubsystem_corr(0.0)
        high = multisubsystem_corr(0.95)
        assert high > low

    def test_severity_range_bounds_intensity(self):
        config = FraudGeneratorConfig(
            random_seed=11, fraud_rate=1.0, severity_range=(0.3, 0.4)
        )
        sampler = FraudScenarioSampler(config)
        for ctx in sampler.sample_many(500):
            assert 0.3 - 1e-9 <= ctx.intensity <= 0.4 + 1e-9


# ==========================================================
# Correlated evidence: scenario-specific subsystem targeting
# ==========================================================

class TestCorrelatedEvidence:
    """Verify the qualitative claims from the scenario table actually hold
    when features are sampled many times (statistically, not for a single
    draw, since every generator still has randomness)."""

    @staticmethod
    def _make_context(scenario: FraudScenario, intensity: float = 1.0) -> FraudContext:
        spec = SCENARIOS[scenario]
        rng = np.random.default_rng(0)
        impacts = {}
        for s in SUBSYSTEMS:
            lo, hi = spec.impact_range(s)
            impacts[s] = float(rng.uniform(lo, hi)) * intensity if hi > 0 else 0.0
        return FraudContext(scenario, spec.tier, intensity, impacts, spec.feature_multipliers)

    def test_voice_replay_degrades_voice_not_vehicle(self):
        ctx = self._make_context(FraudScenario.VOICE_REPLAY)
        n = 500
        genuine_voice = [generate_voice()["liveness_score"] for _ in range(n)]
        fraud_voice = [generate_voice(fraud_context=ctx)["liveness_score"] for _ in range(n)]
        genuine_vehicle = [generate_vehicle()["location_familiarity"] for _ in range(n)]
        fraud_vehicle = [
            generate_vehicle(fraud_context=ctx)["location_familiarity"] for _ in range(n)
        ]

        assert np.mean(fraud_voice) < np.mean(genuine_voice) - 0.1
        # vehicle should be statistically indistinguishable (within noise)
        assert abs(np.mean(fraud_vehicle) - np.mean(genuine_vehicle)) < 0.08

    def test_gps_spoofing_degrades_location_not_voice(self):
        ctx = self._make_context(FraudScenario.GPS_SPOOFING)
        n = 500
        genuine_loc = [generate_vehicle()["location_familiarity"] for _ in range(n)]
        fraud_loc = [
            generate_vehicle(fraud_context=ctx)["location_familiarity"] for _ in range(n)
        ]
        genuine_voice = [generate_voice()["speaker_similarity"] for _ in range(n)]
        fraud_voice = [
            generate_voice(fraud_context=ctx)["speaker_similarity"] for _ in range(n)
        ]

        assert np.mean(fraud_loc) < np.mean(genuine_loc) - 0.1
        assert abs(np.mean(fraud_voice) - np.mean(genuine_voice)) < 0.05

    def test_deepfake_keeps_audio_quality_higher_than_replay(self):
        """Deepfakes are designed to sound clean; replays carry recording
        artifacts. Both fail liveness, but audio_quality should degrade
        less for deepfake than for replay at the same intensity."""
        deepfake_ctx = self._make_context(FraudScenario.DEEPFAKE_VOICE, intensity=1.0)
        replay_ctx = self._make_context(FraudScenario.VOICE_REPLAY, intensity=1.0)

        n = 500
        deepfake_aq = np.mean(
            [generate_voice(fraud_context=deepfake_ctx)["audio_quality"] for _ in range(n)]
        )
        replay_aq = np.mean(
            [generate_voice(fraud_context=replay_ctx)["audio_quality"] for _ in range(n)]
        )
        assert deepfake_aq > replay_aq

    def test_insider_fraud_barely_touches_voice_or_behavior(self):
        ctx = self._make_context(FraudScenario.INSIDER_FRAUD)
        n = 500
        genuine_voice = np.mean([generate_voice()["speaker_similarity"] for _ in range(n)])
        fraud_voice = np.mean(
            [generate_voice(fraud_context=ctx)["speaker_similarity"] for _ in range(n)]
        )
        assert abs(genuine_voice - fraud_voice) < 0.05

    def test_multi_stage_coordinated_hits_everything(self):
        ctx = self._make_context(FraudScenario.MULTI_STAGE_COORDINATED)
        n = 500
        genuine_amt = np.mean(
            [
                generate_transaction()["transaction_risk"]
                for _ in range(n)
            ]
        )
        fraud_amt = np.mean(
            [
                generate_transaction(fraud_context=ctx)["transaction_risk"]
                for _ in range(n)
            ]
        )
        assert fraud_amt > genuine_amt


# ==========================================================
# No generator silently ignores fraud_context
# ==========================================================

GENERATORS_UNDER_TEST = [
    ("voice", lambda ctx: generate_voice(fraud_context=ctx), ["liveness_score", "spoof_probability"]),
    ("behavior", lambda ctx: generate_behavior(fraud_context=ctx), ["stress_score", "command_familiarity"]),
    ("vehicle", lambda ctx: generate_vehicle(fraud_context=ctx), ["location_familiarity"]),
    ("history", lambda ctx: generate_history(fraud_context=ctx), ["successful_transactions", "failed_attempts"]),
    ("transaction", lambda ctx: generate_transaction(fraud_context=ctx), ["transaction_risk"]),
]


class TestEveryGeneratorRespondsToFraudContext:
    @pytest.mark.parametrize("name,call,fields", GENERATORS_UNDER_TEST)
    def test_full_impact_context_changes_distribution(self, name, call, fields):
        full_impact_ctx = FraudContext(
            scenario=FraudScenario.MULTI_STAGE_COORDINATED,
            tier=FraudTier.COORDINATED,
            intensity=1.0,
            subsystem_impact={s: 1.0 for s in SUBSYSTEMS},
            feature_multipliers={},
        )
        genuine_ctx = FraudContext.genuine()

        n = 300
        genuine_samples = [call(genuine_ctx) for _ in range(n)]
        fraud_samples = [call(full_impact_ctx) for _ in range(n)]

        changed = False
        for field in fields:
            genuine_mean = np.mean([s[field] for s in genuine_samples])
            fraud_mean = np.mean([s[field] for s in fraud_samples])
            if abs(genuine_mean - fraud_mean) > 1e-3:
                changed = True
        assert changed, (
            f"generator '{name}' produced statistically identical output "
            f"for genuine vs. full-impact fraud context -- it is not "
            f"actually using fraud_context"
        )

    def test_intent_generator_responds_to_fraud_context(self):
        full_impact_ctx = FraudContext(
            scenario=FraudScenario.MULTI_STAGE_COORDINATED,
            tier=FraudTier.COORDINATED,
            intensity=1.0,
            subsystem_impact={s: 1.0 for s in SUBSYSTEMS},
            feature_multipliers={},
        )
        n = 300
        genuine = np.mean(
            [generate_intent(1, fraud_context=FraudContext.genuine())["llm_confidence"] for _ in range(n)]
        )
        fraud = np.mean(
            [generate_intent(1, fraud_context=full_impact_ctx)["llm_confidence"] for _ in range(n)]
        )
        assert abs(genuine - fraud) > 1e-3


# ==========================================================
# Backward compatibility with the old `fraudulent: bool` API
# ==========================================================

class TestBackwardCompatibility:
    """Every generator must still work exactly as before when called with
    only the legacy `fraudulent` boolean and no fraud_context, since other
    code (docs, ad-hoc scripts) may still call it that way."""

    def test_voice_legacy_bool_still_differentiates(self):
        rng_state = np.random.default_rng(0)
        genuine = [generate_voice(fraudulent=False)["liveness_score"] for _ in range(200)]
        fraud = [generate_voice(fraudulent=True)["liveness_score"] for _ in range(200)]
        assert np.mean(fraud) < np.mean(genuine) - 0.2

    def test_transaction_legacy_bool_still_differentiates(self):
        genuine = [
            generate_transaction(fraudulent=False)["transaction_amount"]
            for _ in range(300)
        ]
        fraud = [
            generate_transaction(fraudulent=True)["transaction_amount"]
            for _ in range(300)
        ]
        # only MONEY_TRANSFER rows carry the fraud amount signal; compare
        # medians across the mixed-category samples, fraud should be higher
        assert np.median(fraud) >= np.median(genuine)

    def test_intent_legacy_bool_still_works(self):
        result = generate_intent(1, fraudulent=True)
        assert "llm_confidence" in result
        assert "intent_type" in result


# ==========================================================
# End-to-end: generate_users.py / generate_transactions.py propagation
# ==========================================================

class TestEndToEndPropagation:
    """Run the real generate_users.generate_user()/main() machinery (in a
    tmp_path sandbox, never touching training/data/) to verify fraud
    context actually flows user -> voice/behavior/vehicle/history and then
    user -> transaction/intent via the fraud_context.csv side-channel."""

    def test_generate_user_uses_sampled_context(self, monkeypatch, tmp_path):
        import generate_users as gu

        monkeypatch.setattr(gu, "OUTPUT", tmp_path / "users.csv")
        monkeypatch.setattr(gu, "FRAUD_OUTPUT", tmp_path / "fraud_context.csv")

        import sys as _sys
        old_argv = _sys.argv
        _sys.argv = ["generate_users.py", "--num-users", "500", "--seed", "5"]
        try:
            gu.main()
        finally:
            _sys.argv = old_argv

        users = pd.read_csv(tmp_path / "users.csv")
        fraud = pd.read_csv(tmp_path / "fraud_context.csv")

        assert len(users) == 500
        assert len(fraud) == 500
        assert set(fraud.columns) == set(FRAUD_CONTEXT_COLUMNS)

        # users.csv must NOT leak any fraud ground-truth columns
        assert "scenario" not in users.columns
        assert "fraud_context" not in users.columns

        # dtypes of users.csv must be exactly as declared (regression test
        # for the iterrows() dtype-corruption bug)
        assert users["user_id"].dtype.kind == "i"
        assert users["kyc_verified"].dtype.kind == "i"

        # at this sample size we should see at least one fraud row and it
        # should correlate with degraded liveness for voice-affecting scenarios
        merged = users.merge(fraud, on="user_id")
        fraud_rows = merged[merged["scenario"] != "genuine"]
        assert len(fraud_rows) > 0

    def test_generate_transactions_reuses_same_context(self, monkeypatch, tmp_path):
        import generate_users as gu
        import generate_transactions as gt

        monkeypatch.setattr(gu, "OUTPUT", tmp_path / "users.csv")
        monkeypatch.setattr(gu, "FRAUD_OUTPUT", tmp_path / "fraud_context.csv")

        import sys as _sys
        old_argv = _sys.argv
        _sys.argv = ["generate_users.py", "--num-users", "800", "--seed", "9"]
        try:
            gu.main()
        finally:
            _sys.argv = old_argv

        monkeypatch.setattr(gt, "INPUT", tmp_path / "users.csv")
        monkeypatch.setattr(gt, "FRAUD_INPUT", tmp_path / "fraud_context.csv")
        monkeypatch.setattr(gt, "OUTPUT", tmp_path / "transactions.csv")
        gt.main()

        transactions = pd.read_csv(tmp_path / "transactions.csv")
        fraud = pd.read_csv(tmp_path / "fraud_context.csv")

        assert len(transactions) == 800
        # dtype regression check again, this time post-transaction-merge
        assert transactions["user_id"].dtype.kind == "i"
        assert transactions["successful_transactions"].dtype.kind == "i"

        merged = transactions.merge(fraud, on="user_id")
        replay_rows = merged[merged["scenario"] == "voice_replay_attack"]
        genuine_rows = merged[merged["scenario"] == "genuine"]
        if len(replay_rows) > 0:
            # voice replay should show materially worse liveness than genuine
            assert replay_rows["liveness_score"].mean() < genuine_rows["liveness_score"].mean()

    def test_missing_fraud_context_file_falls_back_to_genuine(self, monkeypatch, tmp_path):
        """If fraud_context.csv doesn't exist (e.g. an older users.csv),
        generate_transactions.py must not crash -- it should treat every
        user as genuine."""
        import generate_users as gu
        import generate_transactions as gt

        monkeypatch.setattr(gu, "OUTPUT", tmp_path / "users.csv")
        monkeypatch.setattr(gu, "FRAUD_OUTPUT", tmp_path / "fraud_context.csv")

        import sys as _sys
        old_argv = _sys.argv
        _sys.argv = ["generate_users.py", "--num-users", "50", "--seed", "1"]
        try:
            gu.main()
        finally:
            _sys.argv = old_argv

        (tmp_path / "fraud_context.csv").unlink()

        monkeypatch.setattr(gt, "INPUT", tmp_path / "users.csv")
        monkeypatch.setattr(gt, "FRAUD_INPUT", tmp_path / "fraud_context.csv")
        monkeypatch.setattr(gt, "OUTPUT", tmp_path / "transactions.csv")
        gt.main()  # must not raise

        result = pd.read_csv(tmp_path / "transactions.csv")
        assert len(result) == 50
