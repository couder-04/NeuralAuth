"""
explanation.py
===============

Builds explanations for a decision.

Prefers model-native attribution signals — SHAP values, integrated
gradients, or attention weights exposed by the Authentication Network as
`attributions` (preferred) or `feature_attention` (fallback) — over
hardcoded reason strings. Hardcoded strings are only used to supply
context that isn't a model attribution at all (which policy matched, the
risk level, low intent confidence, etc.), or as a last resort when no
attribution signal is available at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .numeric import to_python
from .types import DecisionAction


class ExplanationBuilder:

    @staticmethod
    def top_contributors(authentication, limit: int = 5) -> Dict[str, float]:
        """
        Ranked feature contributions. Looks for, in order of preference:
          1. `attributions`       — e.g. SHAP values / integrated gradients
          2. `feature_attention`  — raw attention weights
        """
        attributions = getattr(authentication, "attributions", None)
        source = attributions if isinstance(attributions, dict) else getattr(
            authentication, "feature_attention", None
        )
        if not isinstance(source, dict):
            return {}

        ranked = sorted(source.items(), key=lambda kv: kv[1], reverse=True)
        return {k: to_python(v) for k, v in ranked[:limit]}

    @staticmethod
    def top_reasons(authentication, risk, policy, intent, limit: int = 5) -> List[str]:
        reasons: List[str] = []

        contributors = ExplanationBuilder.top_contributors(authentication)
        for feature, value in contributors.items():
            label = feature.replace("_", " ").title()
            reasons.append(f"{label} ({value:.2f})")

        matched = getattr(policy, "matched_policy", None) or getattr(policy, "policy_name", None)
        if matched:
            reasons.append(f"Matched policy: {matched}")

        policy_reason = getattr(policy, "reason", None)
        if policy_reason:
            reasons.append(policy_reason)

        risk_level = getattr(risk, "risk_level", None)
        if risk_level:
            reasons.append(f"Risk level: {risk_level}")

        if intent is not None:
            intent_conf = getattr(intent, "confidence", None)
            if intent_conf is not None and intent_conf < 0.5:
                reasons.append("Low intent confidence")

        # Only fall back to hardcoded boolean flags if we found nothing
        # from model attributions at all.
        if not contributors:
            if getattr(risk, "device_known", True) is False:
                reasons.append("Unknown device")
            if getattr(risk, "location_known", True) is False:
                reasons.append("New location detected")
            if getattr(risk, "voice_match_low", False) is True:
                reasons.append("Voice similarity low")

        seen = set()
        unique: List[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique[:limit] or ["No significant risk factors identified"]

    @staticmethod
    def summary(
        action: DecisionAction,
        top_reasons: List[str],
        risk,
        confidence: Optional[float],
        margin: Optional[float] = None,
    ) -> str:
        risk_level = getattr(risk, "risk_level", "UNKNOWN")
        conf_pct = f"{confidence * 100:.0f}%" if confidence is not None else "N/A"
        reason_text = "; ".join(top_reasons[:3]) if top_reasons else "no notable risk factors"

        action_text = {
            DecisionAction.ALLOW: "Transaction approved",
            DecisionAction.VOICE_CHALLENGE: "Voice challenge recommended",
            DecisionAction.OTP: "OTP verification recommended",
            DecisionAction.VOICE_AND_OTP: "Voice and OTP verification recommended",
            DecisionAction.MANUAL_REVIEW: "Sent for manual review",
            DecisionAction.REJECT: "Transaction rejected",
        }[action]

        margin_text = ""
        if margin is not None:
            margin_text = f" Decision margin: {margin:.2f}."

        return (
            f"{action_text}. Risk level: {risk_level}, confidence: {conf_pct}.{margin_text} "
            f"Key factors: {reason_text}."
        )
