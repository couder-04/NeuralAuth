"""
decision_engine
================

Production-grade decision orchestration package for the Transaction
Authentication Engine.

    from decision_engine import DecisionEngine, DecisionConfig, WeightedVoting

    engine = DecisionEngine(
        config=DecisionConfig(),
        strategy=WeightedVoting(),
    )
    result = engine.decide(authentication=..., risk=..., policy=...)
    print(result.to_json())
"""

from .config import DecisionConfig
from .decision_engine import DecisionEngine
from .ensemble import EnsembleAuthentication, combine_ensemble_predictions
from .fusion import (
    BayesianFusion,
    DecisionFusionStrategy,
    FusionContext,
    FusionResult,
    MajorityVoting,
    PolicyFirst,
    RiskFirst,
    RiskWeightedFusion,
    WeightedVoting,
)
from .history import HistoryStore, InMemoryHistoryStore, NullHistoryStore
from .hooks import HookRegistry
from .metrics import MetricsCollector
from .types import DecisionAction, DecisionResult, PolicyPriority, Severity

__all__ = [
    "DecisionEngine",
    "DecisionConfig",
    "DecisionAction",
    "Severity",
    "PolicyPriority",
    "DecisionResult",
    "DecisionFusionStrategy",
    "FusionContext",
    "FusionResult",
    "MajorityVoting",
    "WeightedVoting",
    "RiskWeightedFusion",
    "BayesianFusion",
    "RiskFirst",
    "PolicyFirst",
    "HistoryStore",
    "InMemoryHistoryStore",
    "NullHistoryStore",
    "MetricsCollector",
    "HookRegistry",
    "combine_ensemble_predictions",
    "EnsembleAuthentication",
]