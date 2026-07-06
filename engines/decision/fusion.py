"""
fusion.py
=========

Decision Fusion Strategy pattern.

Instead of one hardcoded `_fuse()` method, every fusion algorithm is a
swappable `DecisionFusionStrategy`:

    engine = DecisionEngine(strategy=WeightedVoting())

Strategies operate on probability *distributions* per source (not just a
single discrete action), because that is how the Authentication Network
and any other model actually reason — and because probabilities are what
let strategies be combined mathematically instead of just picking the
most severe discrete vote.

Included strategies
--------------------
- MajorityVoting   : simple max-severity vote across discrete actions
                      (kept for backwards compatibility / cheap baselines).
- WeightedVoting    : weighted average of each source's probability
                      distribution, with margin-based calibration.
- BayesianFusion    : weighted log-linear (naive-Bayes-style) combination
                      of probability distributions.
- RiskFirst         : hard risk ceiling short-circuits straight to REJECT,
                      otherwise delegates to a fallback strategy.
- PolicyFirst       : policy recommendation always wins (useful as an
                      A/B baseline against the newer strategies).

All strategies respect:
  - the confidence gate (low confidence -> MANUAL_REVIEW)
  - the uncertainty gate (high uncertainty -> at least VOICE_CHALLENGE)
  - policy priority: a CRITICAL policy always wins, regardless of what
    the strategy would otherwise decide.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .types import DecisionAction, PolicyPriority, SEVERITY_RANK


# ============================================================
# Fusion I/O
# ============================================================

@dataclass
class FusionContext:
    votes: Dict[str, DecisionAction]
    probabilities: Dict[str, Dict[str, float]]
    weights: Dict[str, float]
    policy_recommendation: DecisionAction
    policy_priority: PolicyPriority
    confidence: Optional[float]
    uncertainty: Optional[float]
    margin_threshold: float
    confidence_threshold: float
    uncertainty_threshold: float
    overall_risk: Optional[float] = None
    # Per-action linear coefficients {action: (a, b)} used by
    # RiskWeightedFusion to turn `overall_risk` into a probability
    # multiplier: multiplier = b + a * overall_risk. See DecisionConfig.
    risk_bias: Dict[str, Dict[str, float]] = None


@dataclass
class FusionResult:
    action: DecisionAction
    fused_probabilities: Dict[str, float]
    margin: Optional[float]
    override: bool
    override_reason: Optional[str]
    decision_source: str


# ============================================================
# Base strategy
# ============================================================

class DecisionFusionStrategy(ABC):

    name = "base"

    @abstractmethod
    def fuse(self, ctx: FusionContext) -> FusionResult:
        ...

    # ---- shared gates, reused by every concrete strategy -----

    @staticmethod
    def _confidence_gate(ctx: FusionContext) -> Optional[FusionResult]:
        if ctx.confidence is not None and ctx.confidence < ctx.confidence_threshold:
            return FusionResult(
                action=DecisionAction.MANUAL_REVIEW,
                fused_probabilities={DecisionAction.MANUAL_REVIEW.value: 1.0},
                margin=None,
                override=True,
                override_reason="Low model confidence",
                decision_source="confidence_gate",
            )
        return None

    @staticmethod
    def _uncertainty_gate(ctx: FusionContext) -> Optional[FusionResult]:
        if ctx.uncertainty is not None and ctx.uncertainty > ctx.uncertainty_threshold:
            if SEVERITY_RANK[ctx.policy_recommendation] < SEVERITY_RANK[DecisionAction.VOICE_CHALLENGE]:
                return FusionResult(
                    action=DecisionAction.VOICE_CHALLENGE,
                    fused_probabilities={DecisionAction.VOICE_CHALLENGE.value: 1.0},
                    margin=None,
                    override=True,
                    override_reason="High decision uncertainty",
                    decision_source="uncertainty_gate",
                )
        return None

    @staticmethod
    def _apply_policy_priority(
        action: DecisionAction,
        ctx: FusionContext,
    ) -> Optional[FusionResult]:
        """A CRITICAL policy priority always wins, regardless of what the
        strategy otherwise decided. HIGH/MEDIUM/LOW policies do NOT get
        this automatic override — they participate in fusion as a normal
        vote instead."""
        if ctx.policy_priority == PolicyPriority.CRITICAL and action != ctx.policy_recommendation:
            return FusionResult(
                action=ctx.policy_recommendation,
                fused_probabilities={ctx.policy_recommendation.value: 1.0},
                margin=None,
                override=True,
                override_reason="Critical-priority policy overrides fusion result",
                decision_source="policy_priority_critical",
            )
        return None


# ============================================================
# Majority Voting (discrete, severity-based)
# ============================================================

class MajorityVoting(DecisionFusionStrategy):
    """Simple, cheap baseline: the most severe discrete vote wins."""

    name = "majority_voting"

    def fuse(self, ctx: FusionContext) -> FusionResult:
        gate = self._confidence_gate(ctx) or self._uncertainty_gate(ctx)
        if gate:
            return gate

        winner_source, winner_action = max(
            ctx.votes.items(), key=lambda kv: SEVERITY_RANK[kv[1]]
        )

        priority_override = self._apply_policy_priority(winner_action, ctx)
        if priority_override:
            return priority_override

        override = winner_action != ctx.policy_recommendation
        reason = (
            f"'{winner_source}' recommendation ({winner_action.value}) overrides policy"
            if override else None
        )

        return FusionResult(
            action=winner_action,
            fused_probabilities={winner_action.value: 1.0},
            margin=None,
            override=override,
            override_reason=reason,
            decision_source=winner_source,
        )


# ============================================================
# Weighted probabilistic voting
# ============================================================

class WeightedVoting(DecisionFusionStrategy):
    """
    Combines each source's *probability distribution* using configurable
    per-source weights, rather than picking the single most severe
    discrete vote. Includes margin-based calibration: a narrow win
    between the top two candidates escalates the action rather than
    being trusted outright.
    """

    name = "weighted_voting"

    def fuse(self, ctx: FusionContext) -> FusionResult:
        gate = self._confidence_gate(ctx) or self._uncertainty_gate(ctx)
        if gate:
            return gate

        fused = self._weighted_average(ctx.probabilities, ctx.weights)
        action, margin = self._argmax_with_margin(fused)

        override_reason: Optional[str] = None

        if margin is not None and margin < ctx.margin_threshold:
            escalated = self._escalate(action)
            if escalated != action:
                override_reason = f"Low decision margin ({margin:.2f}) triggered escalation"
                action = escalated

        priority_override = self._apply_policy_priority(action, ctx)
        if priority_override:
            return priority_override

        override = action != ctx.policy_recommendation
        if override and override_reason is None:
            override_reason = f"Weighted fusion favored {action.value} over policy"

        return FusionResult(
            action=action,
            fused_probabilities=fused,
            margin=margin,
            override=override,
            override_reason=override_reason,
            decision_source="weighted_voting",
        )

    @staticmethod
    def _weighted_average(
        probabilities: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        weight_sum = 0.0

        for source, dist in probabilities.items():
            w = weights.get(source, weights.get("_default", 0.1))
            weight_sum += w
            for action, p in dist.items():
                totals[action] = totals.get(action, 0.0) + w * p

        if weight_sum <= 0:
            return totals
        return {a: v / weight_sum for a, v in totals.items()}

    @staticmethod
    def _argmax_with_margin(fused: Dict[str, float]) -> Tuple[DecisionAction, Optional[float]]:
        if not fused:
            return DecisionAction.MANUAL_REVIEW, None

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        top_action = DecisionAction(ranked[0][0])
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        return top_action, margin

    @staticmethod
    def _escalate(action: DecisionAction) -> DecisionAction:
        escalation = {
            DecisionAction.ALLOW: DecisionAction.VOICE_CHALLENGE,
            DecisionAction.VOICE_CHALLENGE: DecisionAction.VOICE_AND_OTP,
            DecisionAction.OTP: DecisionAction.VOICE_AND_OTP,
            DecisionAction.VOICE_AND_OTP: DecisionAction.MANUAL_REVIEW,
            DecisionAction.MANUAL_REVIEW: DecisionAction.MANUAL_REVIEW,
            DecisionAction.REJECT: DecisionAction.REJECT,
        }
        return escalation[action]


# ============================================================
# Risk-weighted probabilistic voting
# ============================================================

class RiskWeightedFusion(DecisionFusionStrategy):
    """
    Treats the Risk Engine's `overall_risk` as continuous evidence that
    shifts the fused probability distribution, rather than a threshold
    that short-circuits straight to an action (that's what `RiskFirst`
    used to do, and what the architecture review flagged as wrong: "high
    risk should increase the probability of rejection, not directly force
    rejection").

    Steps:
      1. Weighted-average every source's probability distribution
         (same as WeightedVoting).
      2. Apply a per-action linear risk bias: multiplier = b + a * risk,
         using DecisionConfig.risk_bias_coefficients (e.g. ALLOW gets
         *less* likely as risk rises, REJECT gets *more* likely) --
         equivalent to adding a risk-dependent bias in logit space before
         renormalizing.
      3. Take the argmax of the *risk-adjusted* distribution, with the
         same margin-based escalation as WeightedVoting.

    A hard, unconditional REJECT is deliberately NOT produced here no
    matter how high `overall_risk` gets -- that only happens via a
    CRITICAL-priority policy rule (see
    DecisionFusionStrategy._apply_policy_priority), which is a conscious
    policy decision (e.g. rules/policy_rules.yaml's `RejectCriticalRisk`),
    not an emergent property of one engine's risk score.
    """

    name = "risk_weighted_fusion"

    def fuse(self, ctx: FusionContext) -> FusionResult:
        gate = self._confidence_gate(ctx) or self._uncertainty_gate(ctx)
        if gate:
            return gate

        fused = WeightedVoting._weighted_average(ctx.probabilities, ctx.weights)
        biased = self._apply_risk_bias(fused, ctx.overall_risk, ctx.risk_bias)
        action, margin = WeightedVoting._argmax_with_margin(biased)

        override_reason: Optional[str] = None
        if margin is not None and margin < ctx.margin_threshold:
            escalated = WeightedVoting._escalate(action)
            if escalated != action:
                override_reason = f"Low decision margin ({margin:.2f}) triggered escalation"
                action = escalated

        priority_override = self._apply_policy_priority(action, ctx)
        if priority_override:
            return priority_override

        override = action != ctx.policy_recommendation
        if override and override_reason is None:
            override_reason = f"Risk-weighted fusion favored {action.value} over policy"

        return FusionResult(
            action=action,
            fused_probabilities=biased,
            margin=margin,
            override=override,
            override_reason=override_reason,
            decision_source="risk_weighted_fusion",
        )

    @staticmethod
    def _apply_risk_bias(
        probabilities: Dict[str, float],
        overall_risk: Optional[float],
        coefficients: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[str, float]:
        """multiplier = b + a * overall_risk, applied per action, then
        renormalized. `a > 0` makes an action more likely as risk rises
        (e.g. REJECT, VOICE_AND_OTP); `a < 0` makes it less likely (e.g.
        ALLOW); `a == 0` leaves it unaffected."""
        if overall_risk is None or not coefficients:
            return probabilities

        adjusted: Dict[str, float] = {}
        for action, p in probabilities.items():
            coeff = coefficients.get(action)
            if coeff is None:
                adjusted[action] = p
                continue
            multiplier = max(coeff.get("b", 1.0) + coeff.get("a", 0.0) * overall_risk, 0.0)
            adjusted[action] = p * multiplier

        total = sum(adjusted.values())
        if total <= 0:
            return probabilities
        return {a: v / total for a, v in adjusted.items()}


# ============================================================
# Bayesian (log-linear) fusion
# ============================================================

class BayesianFusion(DecisionFusionStrategy):
    """
    Weighted log-linear combination of probability distributions
    (equivalent to a naive-Bayes-style product-of-experts, normalized
    back into a valid distribution via a softmax over log-scores).
    """

    name = "bayesian_fusion"

    def fuse(self, ctx: FusionContext) -> FusionResult:
        gate = self._confidence_gate(ctx) or self._uncertainty_gate(ctx)
        if gate:
            return gate

        log_scores: Dict[str, float] = {}
        for source, dist in ctx.probabilities.items():
            w = ctx.weights.get(source, ctx.weights.get("_default", 0.1))
            for action, p in dist.items():
                p = max(p, 1e-6)  # avoid log(0)
                log_scores[action] = log_scores.get(action, 0.0) + w * math.log(p)

        if not log_scores:
            fused: Dict[str, float] = {}
            action, margin = DecisionAction.MANUAL_REVIEW, None
        else:
            max_log = max(log_scores.values())
            exp_scores = {a: math.exp(s - max_log) for a, s in log_scores.items()}
            total = sum(exp_scores.values())
            fused = {a: v / total for a, v in exp_scores.items()}
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            action = DecisionAction(ranked[0][0])
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0

        priority_override = self._apply_policy_priority(action, ctx)
        if priority_override:
            return priority_override

        override = action != ctx.policy_recommendation
        reason = f"Bayesian fusion favored {action.value} over policy" if override else None

        return FusionResult(
            action=action,
            fused_probabilities=fused,
            margin=margin,
            override=override,
            override_reason=reason,
            decision_source="bayesian_fusion",
        )


# ============================================================
# Risk-first
# ============================================================

class RiskFirst(DecisionFusionStrategy):
    """
    An extreme, deliberate short-circuit to REJECT -- requires BOTH an
    extreme overall-risk reading AND a CRITICAL policy priority. Neither
    condition alone is sufficient: high risk alone should only shift
    probabilities (see RiskWeightedFusion), never force an outcome by
    itself, and a CRITICAL policy without corroborating risk shouldn't be
    second-guessed by a raw risk number either. Below that joint
    condition, delegates to a fallback strategy (RiskWeightedFusion by
    default).
    """

    name = "risk_first"

    def __init__(
        self,
        fallback: Optional[DecisionFusionStrategy] = None,
        hard_risk_threshold: float = 0.98,
    ):
        self.fallback = fallback or RiskWeightedFusion()
        self.hard_risk_threshold = hard_risk_threshold

    def fuse(self, ctx: FusionContext) -> FusionResult:
        gate = self._confidence_gate(ctx) or self._uncertainty_gate(ctx)
        if gate:
            return gate

        extreme_risk = ctx.overall_risk is not None and ctx.overall_risk >= self.hard_risk_threshold
        critical_policy = ctx.policy_priority == PolicyPriority.CRITICAL

        if extreme_risk and critical_policy:
            action = DecisionAction.REJECT

            override = action != ctx.policy_recommendation
            reason = (
                f"Overall risk ({ctx.overall_risk:.2f}) exceeded the extreme threshold "
                f"while policy priority is CRITICAL"
            )

            return FusionResult(
                action=action,
                fused_probabilities={action.value: 1.0},
                margin=None,
                override=override,
                override_reason=reason,
                decision_source="risk_first",
            )

        result = self.fallback.fuse(ctx)
        result.decision_source = f"risk_first->{result.decision_source}"
        return result


# ============================================================
# Policy-first (legacy baseline)
# ============================================================

class PolicyFirst(DecisionFusionStrategy):
    """Policy recommendation always wins (subject to the confidence /
    uncertainty gates). Kept as an explicit, named strategy so the old
    behavior can still be selected and compared against newer ones."""

    name = "policy_first"

    def fuse(self, ctx: FusionContext) -> FusionResult:
        gate = self._confidence_gate(ctx) or self._uncertainty_gate(ctx)
        if gate:
            return gate

        action = ctx.policy_recommendation
        return FusionResult(
            action=action,
            fused_probabilities={action.value: 1.0},
            margin=None,
            override=False,
            override_reason=None,
            decision_source="policy_first",
        )