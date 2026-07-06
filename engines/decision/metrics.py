"""
metrics.py
==========

Lightweight in-memory metrics collector for production monitoring.
Swap for a real client (statsd, Prometheus, Datadog, etc.) by
implementing the same `record_decision` method and injecting it into
DecisionEngine via `metrics=`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class MetricsCollector:

    def __init__(self):
        self.action_counts: Dict[str, int] = {}
        self.total_decisions: int = 0
        self.total_confidence: float = 0.0
        self.total_latency_ms: float = 0.0
        self.override_count: int = 0
        self.high_uncertainty_count: int = 0

    def record_decision(
        self,
        action: str,
        confidence: Optional[float],
        latency_ms: float,
        policy_override: bool,
        high_uncertainty: bool,
    ) -> None:
        self.total_decisions += 1
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        if confidence is not None:
            self.total_confidence += confidence
        self.total_latency_ms += latency_ms
        if policy_override:
            self.override_count += 1
        if high_uncertainty:
            self.high_uncertainty_count += 1

    def snapshot(self) -> Dict[str, Any]:
        n = max(self.total_decisions, 1)
        return {
            "total_decisions": self.total_decisions,
            "action_counts": dict(self.action_counts),
            "average_confidence": self.total_confidence / n,
            "average_latency_ms": self.total_latency_ms / n,
            "policy_override_rate": self.override_count / n,
            "uncertainty_rate": self.high_uncertainty_count / n,
        }
