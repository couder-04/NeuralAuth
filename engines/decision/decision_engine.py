"""
decision_engine.py
===================

Orchestration layer only. Every other responsibility that used to live
here — fusion algorithm, explanation, audit assembly, metadata, history,
metrics, serialization — now lives in its own module:

    DecisionEngine
        ├── config.py       DecisionConfig            (externalized settings)
        ├── fusion.py        DecisionFusionStrategy    (pluggable voting/fusion)
        ├── explanation.py   ExplanationBuilder        (reasons / summary)
        ├── audit.py         AuditBuilder              (decision trace / graph)
        ├── metadata.py       MetadataBuilder           (request/trace ids)
        ├── history.py        HistoryStore              (external, pluggable)
        ├── metrics.py        MetricsCollector          (monitoring counters)
        ├── hooks.py          HookRegistry              (before/after events)
        ├── ensemble.py       combine_ensemble_predictions
        └── serializers.py    to_json

DecisionEngine's only job is to call each of these in order and assemble
the final DecisionResult.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from .audit import AuditBuilder
from .config import DecisionConfig
from .ensemble import combine_ensemble_predictions
from .explanation import ExplanationBuilder
from .fusion import DecisionFusionStrategy, FusionContext, RiskWeightedFusion
from .history import HistoryStore, InMemoryHistoryStore
from .hooks import HookRegistry
from .metadata import MetadataBuilder
from .metrics import MetricsCollector
from .numeric import to_python
from .types import DecisionAction, DecisionResult, PolicyPriority, SEVERITY_MAP


class DecisionEngine:
    """
    Consumes the outputs of the Authentication Network, Risk Engine,
    Policy Engine, and (optionally) an Intent Engine and any additional
    recommender models, and fuses them into ONE final, explainable,
    auditable decision.
    """

    def __init__(
        self,
        config: Optional[DecisionConfig] = None,
        strategy: Optional[DecisionFusionStrategy] = None,
        history_store: Optional[HistoryStore] = None,
        metrics: Optional[MetricsCollector] = None,
        hooks: Optional[HookRegistry] = None,
    ):
        self.config = config or DecisionConfig()
        self.strategy = strategy or RiskWeightedFusion()
        self.history_store = history_store or InMemoryHistoryStore(maxlen=self.config.history_maxlen)
        self.metrics = metrics or MetricsCollector()
        self.hooks = hooks or HookRegistry()
        self._metadata_builder = MetadataBuilder(self.config.model_version, self.config.policy_version)

    # --------------------------------------------------------
    def decide(
        self,
        authentication: Union[Any, List[Any]],
        risk,
        policy,
        intent=None,
        transaction: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        additional_recommendations: Optional[Dict[str, Any]] = None,
        additional_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> DecisionResult:
        """
        Parameters
        ----------
        authentication
            Output of the Authentication Network, OR a list of outputs
            from an ensemble of models (see ensemble.py) — in which case
            they are merged before proceeding.
        risk
            Output of the Risk Engine.
        policy
            Output of the Policy Engine. May optionally expose a
            `priority` attribute (CRITICAL/HIGH/MEDIUM/LOW); only
            CRITICAL unconditionally overrides fusion.
        intent
            Optional output from the Intent Engine.
        transaction
            Original transaction payload (for metadata / audit / history
            keying).
        request_id
            Optional caller-supplied request id (auto-generated otherwise).
        additional_recommendations
            Optional {source_name: DecisionAction | str} for extra
            recommender models (Fraud AI, Behavior AI, Rule Engine, ...).
        additional_probabilities
            Optional {source_name: {action: probability}} — full
            distributions for the same extra sources, used by the
            probabilistic strategies (WeightedVoting / BayesianFusion).
            If omitted for a source, its vote is treated as one-hot.
        """

        if isinstance(authentication, (list, tuple)):
            authentication = combine_ensemble_predictions(list(authentication))

        timeline: Dict[str, float] = {}
        t_start = time.perf_counter()

        # ---- 1. Votes & probability distributions ----------------
        t0 = time.perf_counter()
        ai_action, ai_probs = self._authentication_recommendation(authentication)
        timeline["authentication_ms"] = self._ms(t0)

        t0 = time.perf_counter()
        policy_action = self._normalize_action(policy.required_action)
        policy_priority = self._normalize_priority(getattr(policy, "priority", None))
        timeline["policy_ms"] = self._ms(t0)

        votes: Dict[str, DecisionAction] = {
            "ai_recommendation": ai_action,
            "policy_recommendation": policy_action,
        }
        probabilities: Dict[str, Dict[str, float]] = {
            "ai_recommendation": ai_probs,
            "policy_recommendation": {policy_action.value: 1.0},
        }

        if additional_recommendations:
            for source, value in additional_recommendations.items():
                action = self._normalize_action(value)
                votes[source] = action
                probabilities[source] = (additional_probabilities or {}).get(
                    source, {action.value: 1.0}
                )

        confidence = to_python(getattr(authentication, "confidence", None))
        uncertainty = to_python(getattr(authentication, "confidence_std", None))
        overall_risk = to_python(getattr(risk, "overall_risk", None))

        # ---- 2. Decision Fusion -----------------------------------
        ctx = FusionContext(
            votes=votes,
            probabilities=probabilities,
            weights=self.config.source_weights,
            policy_recommendation=policy_action,
            policy_priority=policy_priority,
            confidence=confidence,
            uncertainty=uncertainty,
            margin_threshold=self.config.margin_threshold,
            confidence_threshold=self.config.confidence_manual_review_threshold,
            uncertainty_threshold=self.config.uncertainty_voice_challenge_threshold,
            overall_risk=overall_risk,
            risk_bias=self.config.risk_bias_coefficients,
        )

        self.hooks.fire("before_fusion", context=ctx)

        t0 = time.perf_counter()
        fusion_result = self.strategy.fuse(ctx)
        timeline["decision_fusion_ms"] = self._ms(t0)

        self.hooks.fire("after_fusion", result=fusion_result)

        final_action = fusion_result.action
        allowed = final_action == DecisionAction.ALLOW
        voice = final_action in (DecisionAction.VOICE_CHALLENGE, DecisionAction.VOICE_AND_OTP)
        otp = final_action in (DecisionAction.OTP, DecisionAction.VOICE_AND_OTP)
        manual = final_action == DecisionAction.MANUAL_REVIEW
        auth_required = voice or otp

        # ---- 3. Explainability --------------------------------------
        top_reasons = ExplanationBuilder.top_reasons(authentication, risk, policy, intent)
        top_features = ExplanationBuilder.top_contributors(authentication)
        summary = ExplanationBuilder.summary(final_action, top_reasons, risk, confidence, fusion_result.margin)
        message = self._message(final_action)

        # ---- 4. Metadata ----------------------------------------------
        metadata = self._metadata_builder.build(transaction, request_id)

        # ---- 5. History (externalized, pluggable) ----------------------
        history_key = (
            (transaction or {}).get("device_id")
            or (transaction or {}).get("user_id")
            or "global"
        )
        self.history_store.append(history_key, final_action.value)
        decision_history = self.history_store.recent(history_key, self.config.history_maxlen)

        timeline["total_ms"] = self._ms(t_start)

        # ---- 6. Full audit trail / decision graph ------------------------
        self.hooks.fire("before_audit", fusion_result=fusion_result)

        audit = AuditBuilder.build(
            authentication=authentication,
            risk=risk,
            policy=policy,
            intent=intent,
            transaction=transaction,
            fusion_result=fusion_result,
            votes=votes,
            probabilities=probabilities,
            confidence=confidence,
            uncertainty=uncertainty,
            metadata=metadata,
            timeline=timeline,
            top_reasons=top_reasons,
            top_features=top_features,
            decision_history=decision_history,
        )

        self.hooks.fire("after_audit", audit=audit)

        # ---- 7. Metrics ----------------------------------------------------
        self.metrics.record_decision(
            action=final_action.value,
            confidence=confidence,
            latency_ms=timeline["total_ms"],
            policy_override=fusion_result.override,
            high_uncertainty=bool(
                uncertainty is not None and uncertainty > self.config.uncertainty_voice_challenge_threshold
            ),
        )

        severity = SEVERITY_MAP[final_action]

        return DecisionResult(
            status="SUCCESS",
            decision_trace_id=metadata["decision_trace_id"],
            request_id=metadata["request_id"],
            model_version=metadata["model_version"],
            policy_version=metadata["policy_version"],
            latency_ms=timeline["total_ms"],
            decision_source=fusion_result.decision_source,
            margin=fusion_result.margin,
            action=final_action,
            severity=severity,
            transaction_allowed=allowed,
            authentication_required=auth_required,
            voice_required=voice,
            otp_required=otp,
            manual_review=manual,
            confidence=confidence,
            message=message,
            reason=getattr(policy, "reason", "") or "",
            summary=summary,
            policy_override=fusion_result.override,
            override_reason=fusion_result.override_reason,
            top_reasons=top_reasons,
            decision_probabilities=fusion_result.fused_probabilities,
            audit_log=audit,
        )

    # ============================================================
    # Authentication recommendation
    # ============================================================

    @staticmethod
    def _authentication_recommendation(authentication):
        """
        Use the Authentication Network's own predicted probability
        distribution (argmax over `decision_probabilities`) rather than
        an invented heuristic — this guarantees the network and the
        decision engine can never disagree about what the network itself
        predicted. Falls back to a single `recommended_action` field, and
        only to a coarse heuristic if the network exposes neither.
        """

        probs = getattr(authentication, "decision_probabilities", None)
        if isinstance(probs, dict) and probs:
            top_action = max(probs.items(), key=lambda kv: kv[1])[0]
            return DecisionAction(top_action), {k: to_python(v) for k, v in probs.items()}

        recommendation = getattr(authentication, "recommended_action", None)
        if recommendation is not None:
            action = DecisionEngine._normalize_action(recommendation)
            return action, {action.value: 1.0}

        # Last-resort heuristic — only used when the network exposes
        # neither a probability distribution nor a discrete recommendation.
        trust = to_python(getattr(authentication, "trust_score", None)) or 0.0
        risk_score = to_python(getattr(authentication, "risk_score", None)) or 0.0

        if risk_score >= 0.75:
            action = DecisionAction.REJECT
        elif risk_score >= 0.5:
            action = DecisionAction.VOICE_AND_OTP
        elif risk_score >= 0.3:
            action = DecisionAction.VOICE_CHALLENGE
        elif trust >= 0.6:
            action = DecisionAction.ALLOW
        else:
            action = DecisionAction.OTP

        return action, {action.value: 1.0}

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _normalize_action(value) -> DecisionAction:
        raw = value.value if hasattr(value, "value") else value
        return DecisionAction(raw)

    @staticmethod
    def _normalize_priority(value) -> PolicyPriority:
        if value is None:
            return PolicyPriority.MEDIUM
        raw = value.value if hasattr(value, "value") else value
        try:
            return PolicyPriority(raw)
        except ValueError:
            return PolicyPriority.MEDIUM

    @staticmethod
    def _message(action: DecisionAction) -> str:
        messages = {
            DecisionAction.ALLOW: "Transaction Approved.",
            DecisionAction.VOICE_CHALLENGE: "Voice verification required.",
            DecisionAction.OTP: "OTP verification required.",
            DecisionAction.VOICE_AND_OTP: "Voice verification and OTP required.",
            DecisionAction.MANUAL_REVIEW: "Transaction sent for manual review.",
            DecisionAction.REJECT: "Transaction rejected.",
        }
        return messages[action]

    @staticmethod
    def _ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 3)