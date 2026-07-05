"""
policy_engine.py
=================

The Policy Engine is the business-rules layer of the authentication system.

It performs NO machine learning. It receives predictions from the
Authentication Network, Intent Engine, and Risk Engine, combines them with
transaction context, and converts them into a deterministic security policy.

    AI says what it believes.
    Policy Engine decides what the bank allows.

Design principles
------------------
* Deterministic  - same input -> same output
* Explainable    - every decision has a reason
* Configurable   - YAML-driven, no code changes for new policies
* Auditable      - logs every matched rule
* Extensible     - new policy types via YAML only
* Independent    - no ML inside this module
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for policy_engine.py. Install with `pip install pyyaml`."
    ) from exc


logger = logging.getLogger("policy_engine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | policy_engine | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


DEFAULT_RULES_PATH = Path(__file__).parent / "rules" / "policy_rules.yaml"


# ---------------------------------------------------------------------------
# PolicyAction
# ---------------------------------------------------------------------------

class PolicyAction(Enum):
    """Possible actions the Policy Engine can require."""

    ALLOW = "ALLOW"
    VOICE_CHALLENGE = "VOICE_CHALLENGE"
    OTP = "OTP"
    VOICE_AND_OTP = "VOICE_AND_OTP"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"

    @property
    def severity(self) -> int:
        """Higher severity wins when resolving conflicts between rules."""
        order = {
            PolicyAction.ALLOW: 0,
            PolicyAction.VOICE_CHALLENGE: 1,
            PolicyAction.OTP: 2,
            PolicyAction.VOICE_AND_OTP: 3,
            PolicyAction.MANUAL_REVIEW: 4,
            PolicyAction.REJECT: 5,
        }
        return order[self]


# Evaluation order used to group rules (independent of severity, used for
# organizing/reporting only -- the actual winner is chosen by priority then
# severity, see resolve_conflicts()).
EVALUATION_ORDER = [
    PolicyAction.REJECT,
    PolicyAction.MANUAL_REVIEW,
    PolicyAction.VOICE_AND_OTP,
    PolicyAction.OTP,
    PolicyAction.VOICE_CHALLENGE,
    PolicyAction.ALLOW,
]


# ---------------------------------------------------------------------------
# PolicyInput
# ---------------------------------------------------------------------------

@dataclass
class PolicyInput:
    """Everything the Policy Engine needs to make a decision."""

    # From Authentication Network
    trust_score: float
    risk_score: float
    confidence: float
    network_decision: str

    # From Intent Engine
    intent: str
    intent_confidence: float

    # From Risk Engine
    risk_level: str  # e.g. "LOW", "MEDIUM", "HIGH", "CRITICAL"

    # Transaction / context fields
    transaction_amount: float = 0.0
    beneficiary_type: str = "KNOWN"          # e.g. "KNOWN", "NEW"
    location_familiarity: str = "FAMILIAR"   # e.g. "FAMILIAR", "UNFAMILIAR"
    time_familiarity: str = "NORMAL"         # e.g. "NORMAL", "ODD_HOUR"
    previous_trust_score: Optional[float] = None
    failed_attempts: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# PolicyRule
# ---------------------------------------------------------------------------

# Maps a YAML condition-key suffix to a comparison operator.
_OPERATOR_SUFFIXES: dict[str, Callable[[Any, Any], bool]] = {
    "_lt": operator.lt,
    "_lte": operator.le,
    "_gt": operator.gt,
    "_gte": operator.ge,
    "_eq": operator.eq,
    "_ne": operator.ne,
    "_in": lambda value, options: value in options,
}


@dataclass
class PolicyRule:
    """A single, declarative rule loaded from YAML."""

    name: str
    priority: int
    action: PolicyAction
    when: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    requires_voice: bool = False
    requires_otp: bool = False
    requires_manual_review: bool = False
    override_network_decision: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> "PolicyRule":
        try:
            action = PolicyAction(raw["action"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Rule '{raw.get('name', '<unnamed>')}' has an invalid or "
                f"missing action: {raw.get('action')!r}"
            ) from exc

        return cls(
            name=raw["name"],
            priority=int(raw.get("priority", 0)),
            action=action,
            when=raw.get("when", {}) or {},
            reason=raw.get("reason"),
            requires_voice=action in (PolicyAction.VOICE_CHALLENGE, PolicyAction.VOICE_AND_OTP),
            requires_otp=action in (PolicyAction.OTP, PolicyAction.VOICE_AND_OTP),
            requires_manual_review=action == PolicyAction.MANUAL_REVIEW,
            override_network_decision=bool(raw.get("override_network_decision", True)),
        )

    def matches(self, ctx: dict) -> bool:
        """Return True if every condition in `when` is satisfied by ctx."""
        for key, expected in self.when.items():
            field_name, op_func = _resolve_condition_key(key)
            if field_name not in ctx:
                logger.warning(
                    "Rule '%s' references unknown field '%s'; skipping condition.",
                    self.name,
                    field_name,
                )
                return False
            actual = ctx[field_name]
            try:
                if not op_func(actual, expected):
                    return False
            except TypeError:
                # Type mismatch (e.g. comparing str with a numeric operator)
                return False
        return True


def _resolve_condition_key(key: str) -> tuple[str, Callable[[Any, Any], bool]]:
    """Split a YAML condition key like 'trust_score_lt' into
    ('trust_score', operator.lt). Defaults to equality if no suffix matches.
    """
    for suffix, op_func in _OPERATOR_SUFFIXES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], op_func
    return key, operator.eq


# ---------------------------------------------------------------------------
# PolicyResult
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    """The final decision returned by the Policy Engine."""

    required_action: PolicyAction
    policy_name: str
    reason: str
    policy_priority: int
    requires_voice: bool
    requires_otp: bool
    requires_manual_review: bool
    override_network_decision: bool
    audit_message: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_action"] = self.required_action.value
        return d


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Deterministic rules engine that converts model predictions into a
    required security policy.
    """

    def __init__(self, rules_path: Optional[str | Path] = None, auto_load: bool = True):
        self.rules_path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
        self.rules: list[PolicyRule] = []
        if auto_load:
            self.load_rules()

    # -- Loading / validation -----------------------------------------

    def load_rules(self) -> None:
        """Load and validate rules from the configured YAML file."""
        if not self.rules_path.exists():
            logger.warning(
                "Rules file not found at %s; falling back to built-in defaults.",
                self.rules_path,
            )
            raw_rules = _DEFAULT_RULES
        else:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                raw_rules = yaml.safe_load(f) or []

        rules = [PolicyRule.from_dict(r) for r in raw_rules]
        self.validate_rules(rules)
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        logger.info("Loaded %d policy rules from %s.", len(self.rules), self.rules_path)

    def reload_yaml(self) -> None:
        """Hot-reload rules from disk without restarting the service."""
        logger.info("Reloading policy rules from %s.", self.rules_path)
        self.load_rules()

    def validate_rules(self, rules: list[PolicyRule]) -> None:
        """Basic sanity checks: unique names, non-negative priorities."""
        seen_names = set()
        for rule in rules:
            if rule.name in seen_names:
                raise ValueError(f"Duplicate rule name detected: '{rule.name}'")
            seen_names.add(rule.name)
            if rule.priority < 0:
                raise ValueError(f"Rule '{rule.name}' has a negative priority.")
            if not rule.when:
                logger.warning(
                    "Rule '%s' has no conditions and will match every input.",
                    rule.name,
                )

    # -- Evaluation ------------------------------------------------------

    def evaluate(self, policy_input: PolicyInput) -> PolicyResult:
        """Evaluate all rules against policy_input and return the winning
        PolicyResult, resolving any conflicts by priority then severity.
        """
        ctx = policy_input.as_dict()
        matched = self.match_rule(ctx)

        if not matched:
            return self._allow_result(policy_input)

        winner = self.resolve_conflicts(matched)
        audit = self.generate_audit(policy_input, winner, matched)

        result = PolicyResult(
            required_action=winner.action,
            policy_name=winner.name,
            reason=winner.reason or f"Matched rule '{winner.name}'.",
            policy_priority=winner.priority,
            requires_voice=winner.requires_voice,
            requires_otp=winner.requires_otp,
            requires_manual_review=winner.requires_manual_review,
            override_network_decision=(
                winner.override_network_decision
                and winner.action.value != policy_input.network_decision
            ),
            audit_message=audit,
        )
        logger.info(
            "Decision=%s policy=%s priority=%d network_decision=%s override=%s",
            result.required_action.value,
            result.policy_name,
            result.policy_priority,
            policy_input.network_decision,
            result.override_network_decision,
        )
        return result

    def evaluate_rule(self, rule: PolicyRule, ctx: dict) -> bool:
        """Evaluate a single rule against a context dict."""
        return rule.matches(ctx)

    def match_rule(self, ctx: dict) -> list[PolicyRule]:
        """Return every rule whose conditions are satisfied by ctx, in
        priority order (highest first).
        """
        return [rule for rule in self.rules if self.evaluate_rule(rule, ctx)]

    def resolve_conflicts(self, matched: list[PolicyRule]) -> PolicyRule:
        """Pick the winning rule among all matches.

        Resolution order:
          1. Highest priority wins.
          2. Ties broken by highest action severity (REJECT > MANUAL_REVIEW
             > VOICE_AND_OTP > OTP > VOICE_CHALLENGE > ALLOW).
        """
        return max(matched, key=lambda r: (r.priority, r.action.severity))

    # -- Audit / explainability ------------------------------------------

    def generate_audit(
        self,
        policy_input: PolicyInput,
        winner: PolicyRule,
        matched: list[PolicyRule],
    ) -> dict:
        """Build a structured, explainable audit record for this decision."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": winner.action.value,
            "matched_policy": winner.name,
            "reason": winner.reason or f"Matched rule '{winner.name}'.",
            "network_decision": policy_input.network_decision,
            "risk_level": policy_input.risk_level,
            "override": (
                winner.override_network_decision
                and winner.action.value != policy_input.network_decision
            ),
            "other_matched_policies": [
                r.name for r in matched if r.name != winner.name
            ],
        }

    def export_policy(self, result: PolicyResult) -> dict:
        """Serialize a PolicyResult for downstream consumers (e.g. the
        Decision Engine) or for logging/telemetry pipelines.
        """
        return result.to_dict()

    # -- Internal helpers --------------------------------------------------

    def _allow_result(self, policy_input: PolicyInput) -> PolicyResult:
        """Default result when no rule matches: fall back to ALLOW, but
        never override the network's own decision.
        """
        audit = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": PolicyAction.ALLOW.value,
            "matched_policy": "DefaultAllow",
            "reason": "No policy rule matched; defaulting to network decision.",
            "network_decision": policy_input.network_decision,
            "risk_level": policy_input.risk_level,
            "override": False,
            "other_matched_policies": [],
        }
        logger.info("No rule matched; defaulting to ALLOW (network decision preserved).")
        return PolicyResult(
            required_action=PolicyAction.ALLOW,
            policy_name="DefaultAllow",
            reason="No policy rule matched; defaulting to network decision.",
            policy_priority=0,
            requires_voice=False,
            requires_otp=False,
            requires_manual_review=False,
            override_network_decision=False,
            audit_message=audit,
        )


# ---------------------------------------------------------------------------
# Built-in default rules (used only if policy_rules.yaml is missing)
# ---------------------------------------------------------------------------

_DEFAULT_RULES: list[dict] = [
    {
        "name": "RejectCriticalRisk",
        "priority": 100,
        "when": {"risk_level_eq": "CRITICAL"},
        "action": "REJECT",
        "reason": "Risk Engine flagged this transaction as CRITICAL risk.",
    },
    {
        "name": "RejectLowTrust",
        "priority": 100,
        "when": {"trust_score_lt": 0.2},
        "action": "REJECT",
        "reason": "Trust score is below the minimum acceptable threshold.",
    },
    {
        "name": "ManualReviewRepeatedFailures",
        "priority": 90,
        "when": {"failed_attempts_gte": 3},
        "action": "MANUAL_REVIEW",
        "reason": "Three or more failed authentication attempts.",
    },
    {
        "name": "HighRiskTransaction",
        "priority": 80,
        "when": {"risk_level_eq": "HIGH"},
        "action": "VOICE_AND_OTP",
        "reason": "High transaction risk requires strong authentication.",
    },
    {
        "name": "LowConfidencePrediction",
        "priority": 70,
        "when": {"confidence_lt": 0.4},
        "action": "VOICE_AND_OTP",
        "reason": "Authentication Network confidence is too low to trust alone.",
    },
    {
        "name": "NewBeneficiary",
        "priority": 60,
        "when": {"beneficiary_type_eq": "NEW"},
        "action": "VOICE_AND_OTP",
        "reason": "Transfers to a new beneficiary require strong authentication.",
    },
    {
        "name": "LargeTransaction",
        "priority": 50,
        "when": {"transaction_amount_gt": 100000},
        "action": "OTP",
        "reason": "Amount exceeds the bank's standard threshold.",
    },
    {
        "name": "UnfamiliarLocation",
        "priority": 40,
        "when": {"location_familiarity_eq": "UNFAMILIAR"},
        "action": "OTP",
        "reason": "Transaction originates from an unfamiliar location.",
    },
    {
        "name": "LowIntentConfidence",
        "priority": 30,
        "when": {"intent_confidence_lt": 0.6},
        "action": "VOICE_CHALLENGE",
        "reason": "Intent Engine is not confident about the transaction purpose.",
    },
    {
        "name": "OddHourActivity",
        "priority": 20,
        "when": {"time_familiarity_eq": "ODD_HOUR"},
        "action": "VOICE_CHALLENGE",
        "reason": "Transaction occurs at an unusual hour for this user.",
    },
    {
        "name": "TrustedUser",
        "priority": 10,
        "when": {"trust_score_gte": 0.9},
        "action": "ALLOW",
        "reason": "High trust score; no additional friction required.",
    },
]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = PolicyEngine()  # loads rules/policy_rules.yaml if present

    sample_input = PolicyInput(
        trust_score=0.55,
        risk_score=0.72,
        confidence=0.61,
        network_decision="ALLOW",
        intent="WIRE_TRANSFER",
        intent_confidence=0.58,
        risk_level="HIGH",
        transaction_amount=15000.0,
        beneficiary_type="NEW",
        location_familiarity="FAMILIAR",
        time_familiarity="NORMAL",
        failed_attempts=0,
    )

    decision = engine.evaluate(sample_input)
    import json

    print(json.dumps(engine.export_policy(decision), indent=2))