"""
types.py
========

Shared enums and the DecisionResult data contract. Kept dependency-free
(no imports from other decision_engine modules) so every other module can
import from here without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Final Actions
# ============================================================

class DecisionAction(str, Enum):
    ALLOW = "ALLOW"
    VOICE_CHALLENGE = "VOICE_CHALLENGE"
    OTP = "OTP"
    VOICE_AND_OTP = "VOICE_AND_OTP"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PolicyPriority(str, Enum):
    """
    Policy priority tiers. Only CRITICAL unconditionally overrides the
    fusion result — HIGH/MEDIUM/LOW policies participate in fusion like
    any other vote instead of automatically winning.
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


SEVERITY_MAP: Dict[DecisionAction, Severity] = {
    DecisionAction.ALLOW: Severity.INFO,
    DecisionAction.VOICE_CHALLENGE: Severity.WARNING,
    DecisionAction.OTP: Severity.WARNING,
    DecisionAction.VOICE_AND_OTP: Severity.WARNING,
    DecisionAction.MANUAL_REVIEW: Severity.WARNING,
    DecisionAction.REJECT: Severity.CRITICAL,
}

# How "strict" each action is — used by majority-voting / gating logic,
# NOT by the probabilistic strategies (which combine probabilities
# directly instead of ranking discrete actions).
SEVERITY_RANK: Dict[DecisionAction, int] = {
    DecisionAction.ALLOW: 0,
    DecisionAction.VOICE_CHALLENGE: 1,
    DecisionAction.OTP: 1,
    DecisionAction.VOICE_AND_OTP: 2,
    DecisionAction.MANUAL_REVIEW: 3,
    DecisionAction.REJECT: 4,
}


# ============================================================
# Final Output
# ============================================================

@dataclass
class DecisionResult:

    status: str

    # Top-level trace/versioning fields (point: don't bury these in audit).
    decision_trace_id: str
    request_id: str
    model_version: str
    policy_version: str
    latency_ms: float
    decision_source: str
    margin: Optional[float]

    action: DecisionAction
    severity: Severity

    transaction_allowed: bool
    authentication_required: bool
    voice_required: bool
    otp_required: bool
    manual_review: bool

    confidence: Optional[float]
    message: str
    reason: str
    summary: str

    policy_override: bool
    override_reason: Optional[str]

    top_reasons: List[str]
    decision_probabilities: Dict[str, float]

    audit_log: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Machine-readable payload. Delegates to serializers.to_json
        (imported lazily to avoid a circular import at module load time)."""
        from .serializers import to_json as _to_json
        return _to_json(self)
