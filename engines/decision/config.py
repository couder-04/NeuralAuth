"""
config.py
=========

Externalizes everything that was previously a hardcoded module-level
constant (thresholds, versions, source weights) into a single
configurable object. Can be constructed directly, or loaded from a
config.yaml for ops teams to tune without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DecisionConfig:

    # Confidence below this -> forced MANUAL_REVIEW.
    confidence_manual_review_threshold: float = 0.35

    # Uncertainty (confidence_std) above this -> escalate to at least
    # VOICE_CHALLENGE.
    uncertainty_voice_challenge_threshold: float = 0.25

    # Minimum gap between the top-1 and top-2 fused probabilities before
    # the decision is considered well-calibrated. Below this, the
    # probabilistic strategies escalate rather than trust a narrow win.
    margin_threshold: float = 0.15

    # Hard risk ceiling used by the RiskFirst strategy.
    hard_risk_reject_threshold: float = 0.70

    model_version: str = "auth-net-v3.2.1"
    policy_version: str = "policy-v1.4.0"

    history_maxlen: int = 10

    # Per-source weights consumed by WeightedVoting / BayesianFusion.
    # "_default" is used for any source not explicitly listed (e.g. a
    # newly registered model that hasn't been tuned yet).
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "ai_recommendation": 0.45,
        "fraud_ai": 0.30,
        "behavior_ai": 0.15,
        "policy_recommendation": 0.10,
        "_default": 0.10,
    })

    # Per-action linear risk-bias coefficients consumed by
    # RiskWeightedFusion: multiplier = b + a * overall_risk, applied to
    # that action's fused probability before renormalizing and taking
    # the argmax. `a > 0` makes an action MORE likely as risk rises
    # (VOICE_AND_OTP, REJECT); `a < 0` makes it LESS likely (ALLOW);
    # VOICE_CHALLENGE is left unbiased (a=0) so it acts as the "neutral"
    # middle ground. This is what lets high risk shift probability mass
    # toward stronger authentication/rejection instead of hard-forcing a
    # single outcome (see architecture review).
    risk_bias_coefficients: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "ALLOW": {"a": -1.0, "b": 1.0},
        "VOICE_CHALLENGE": {"a": 0.0, "b": 1.0},
        "OTP": {"a": 0.5, "b": 1.0},
        "VOICE_AND_OTP": {"a": 1.0, "b": 1.0},
        "MANUAL_REVIEW": {"a": 0.5, "b": 1.0},
        "REJECT": {"a": 2.0, "b": 1.0},
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "DecisionConfig":
        """
        Load configuration from a config.yaml, e.g.:

            confidence_manual_review_threshold: 0.35
            uncertainty_voice_challenge_threshold: 0.25
            margin_threshold: 0.15
            hard_risk_reject_threshold: 0.70
            model_version: auth-net-v3.2.1
            policy_version: policy-v1.4.0
            history_maxlen: 10
            source_weights:
              ai_recommendation: 0.45
              fraud_ai: 0.30
              behavior_ai: 0.15
              policy_recommendation: 0.10
              _default: 0.10

        Requires PyYAML (`pip install pyyaml`) — imported lazily so the
        rest of the package has no hard dependency on it.
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def weight_for(self, source: str) -> float:
        return self.source_weights.get(source, self.source_weights.get("_default", 0.1))