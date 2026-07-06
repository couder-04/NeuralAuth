"""
serializers.py
==============

Machine-readable (JSON / API) serialization of a DecisionResult, kept
separate from DecisionResult itself so output formats can evolve (e.g.
adding a v2 payload shape) without touching the data contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .types import DecisionResult


def to_json(result: "DecisionResult") -> Dict[str, Any]:
    return {
        "decision_trace_id": result.decision_trace_id,
        "request_id": result.request_id,
        "model_version": result.model_version,
        "policy_version": result.policy_version,
        "decision_source": result.decision_source,
        "latency_ms": result.latency_ms,
        "decision": result.action.value,
        "severity": result.severity.value,
        "confidence": result.confidence,
        "margin": result.margin,
        "transaction_allowed": result.transaction_allowed,
        "authentication_required": result.authentication_required,
        "voice_required": result.voice_required,
        "otp_required": result.otp_required,
        "manual_review": result.manual_review,
        "policy_override": result.policy_override,
        "override_reason": result.override_reason,
        "reasons": result.top_reasons,
        "decision_probabilities": result.decision_probabilities,
        "message": result.message,
        "summary": result.summary,
        "audit": result.audit_log,
    }
