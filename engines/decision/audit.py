"""
audit.py
========

Builds the full decision trace / audit payload, plus a linear
"decision graph" (Authentication -> Risk -> Policy -> Fusion -> Decision)
that's convenient for dashboards to render directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .numeric import to_python


class AuditBuilder:

    @staticmethod
    def risk_breakdown(risk) -> Dict[str, float]:
        breakdown = getattr(risk, "breakdown", None)
        if isinstance(breakdown, dict):
            return {k: to_python(v) for k, v in breakdown.items()}

        fields = ("behavior_risk", "voice_risk", "transaction_risk", "device_risk", "location_risk")
        result: Dict[str, float] = {}
        for f in fields:
            v = getattr(risk, f, None)
            if v is not None:
                result[f.replace("_risk", "")] = to_python(v)
        return result

    @staticmethod
    def rule_trace(policy) -> List[Dict[str, Any]]:
        rules = getattr(policy, "rule_trace", None) or getattr(policy, "matched_rules", None)
        if not rules:
            return []

        trace = []
        for rule in rules:
            if isinstance(rule, dict):
                trace.append({
                    "rule": rule.get("name", rule.get("rule", "unknown")),
                    "passed": rule.get("passed", rule.get("status") == "PASSED"),
                })
            else:
                trace.append({"rule": str(rule), "passed": True})
        return trace

    @staticmethod
    def decision_graph(fusion_result, votes) -> List[Dict[str, Any]]:
        """Linear stage-by-stage graph, convenient for dashboards:
        Authentication -> Policy -> Fusion -> Decision."""
        return [
            {"stage": "authentication", "output": votes.get("ai_recommendation").value
                if votes.get("ai_recommendation") else None},
            {"stage": "policy", "output": votes.get("policy_recommendation").value
                if votes.get("policy_recommendation") else None},
            {"stage": "fusion", "strategy": fusion_result.decision_source, "output": fusion_result.action.value},
            {"stage": "decision", "output": fusion_result.action.value},
        ]

    @staticmethod
    def build(
        *,
        authentication,
        risk,
        policy,
        intent,
        transaction,
        fusion_result,
        votes,
        probabilities,
        confidence,
        uncertainty,
        metadata: Dict[str, Any],
        timeline: Dict[str, float],
        top_reasons: List[str],
        top_attributions: Dict[str, float],
        decision_history: List[str],
        feature_vector: Dict[str, Any] = None,
    ) -> Dict[str, Any]:

        audit: Dict[str, Any] = {
            "metadata": metadata,
            "decision_trace": {
                "authentication": {
                    "trust_score": to_python(getattr(authentication, "trust_score", None)),
                    "risk_score": to_python(getattr(authentication, "risk_score", None)),
                    "confidence": confidence,
                    "confidence_std": uncertainty,
                    "decision_probabilities": probabilities.get("ai_recommendation"),
                },
                "risk": {
                    "overall_risk": to_python(getattr(risk, "overall_risk", None)),
                    "risk_level": getattr(risk, "risk_level", None),
                    "breakdown": AuditBuilder.risk_breakdown(risk),
                },
                "policy": {
                    "matched_policy": getattr(policy, "matched_policy", None) or getattr(policy, "policy_name", None),
                    "priority": getattr(policy, "priority", None),
                    "required_action": votes["policy_recommendation"].value,
                    "reason": getattr(policy, "reason", None),
                    "rule_trace": AuditBuilder.rule_trace(policy),
                },
                "fusion": {
                    "strategy": fusion_result.decision_source,
                    "votes": {source: action.value for source, action in votes.items()},
                    "fused_probabilities": fusion_result.fused_probabilities,
                    "margin": fusion_result.margin,
                    "final_action": fusion_result.action.value,
                    "policy_override": fusion_result.override,
                    "override_reason": fusion_result.override_reason,
                },
            },
            "decision_graph": AuditBuilder.decision_graph(fusion_result, votes),
            "top_reasons": top_reasons,
            # Top-N model attribution/attention scores (SHAP/integrated
            # gradients/attention weights) -- NOT the engineered feature
            # vector. Named `top_attributions` (not `top_features`) so it
            # can't be confused with -- or accidentally keyword-matched
            # in place of -- the actual `feature_vector` key below (see
            # dashboard.py::find_dict_by_keywords, which used to match
            # this dict by mistake because its old name, `top_features`,
            # contained the substring "features").
            "top_attributions": top_attributions,
            "timeline_ms": timeline,
            "decision_history": decision_history,
        }

        if intent is not None:
            audit["decision_trace"]["intent"] = {
                "intent": getattr(intent, "intent", None),
                "confidence": to_python(getattr(intent, "confidence", None)),
            }

        if transaction is not None:
            audit["transaction"] = transaction

        if feature_vector is not None:
            # The full set of engineered features (see
            # engines/feature_extractor.py / models/feature_vector.py)
            # that fed the Authentication Network / Risk Engine / Policy
            # Engine for this request -- included verbatim so dashboards
            # and offline analysis can inspect every input signal, not
            # just the top-N attribution scores above.
            audit["feature_vector"] = {k: to_python(v) for k, v in feature_vector.items()}

        return audit
