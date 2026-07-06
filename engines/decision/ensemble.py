"""
ensemble.py
===========

Ensemble support: the Authentication step can supply a single prediction
OR a `List[AuthenticationPrediction]` (e.g. from a deep ensemble / MC
Dropout with multiple forward passes materialized as separate objects).
`combine_ensemble_predictions` merges them into one pseudo-authentication
object that the rest of the pipeline consumes exactly like a single
model's output.

Disagreement across ensemble members is itself a signal: when individual
confidences diverge, that divergence is folded into `confidence_std` so
the uncertainty gate can see it even if a single member reported high
confidence.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional

from .numeric import to_python


class EnsembleAuthentication:
    """Merged pseudo-authentication-network output."""

    def __init__(
        self,
        trust_score: Optional[float],
        risk_score: Optional[float],
        confidence: Optional[float],
        confidence_std: Optional[float],
        decision_probabilities: Optional[Dict[str, float]],
        feature_attention: Optional[Dict[str, float]],
        recommended_action: Optional[Any] = None,
    ):
        self.trust_score = trust_score
        self.risk_score = risk_score
        self.confidence = confidence
        self.confidence_std = confidence_std
        self.decision_probabilities = decision_probabilities
        self.feature_attention = feature_attention
        self.recommended_action = recommended_action


def combine_ensemble_predictions(predictions: List[Any]) -> EnsembleAuthentication:
    if not predictions:
        raise ValueError("combine_ensemble_predictions requires at least one prediction")

    if len(predictions) == 1:
        p = predictions[0]
        return EnsembleAuthentication(
            trust_score=to_python(getattr(p, "trust_score", None)),
            risk_score=to_python(getattr(p, "risk_score", None)),
            confidence=to_python(getattr(p, "confidence", None)),
            confidence_std=to_python(getattr(p, "confidence_std", None)),
            decision_probabilities=getattr(p, "decision_probabilities", None),
            feature_attention=getattr(p, "feature_attention", None),
            recommended_action=getattr(p, "recommended_action", None),
        )

    trust_scores = [to_python(getattr(p, "trust_score", None)) for p in predictions]
    trust_scores = [v for v in trust_scores if v is not None]

    risk_scores = [to_python(getattr(p, "risk_score", None)) for p in predictions]
    risk_scores = [v for v in risk_scores if v is not None]

    confidences = [to_python(getattr(p, "confidence", None)) for p in predictions]
    confidences = [v for v in confidences if v is not None]

    # Disagreement across ensemble members is itself an uncertainty signal.
    confidence_std = None
    if len(confidences) > 1:
        avg = mean(confidences)
        variance = mean([(c - avg) ** 2 for c in confidences])
        confidence_std = variance ** 0.5

    prob_totals: Dict[str, float] = {}
    prob_count = 0
    for p in predictions:
        dist = getattr(p, "decision_probabilities", None)
        if isinstance(dist, dict):
            prob_count += 1
            for action, prob in dist.items():
                prob_totals[action] = prob_totals.get(action, 0.0) + prob
    fused_probs = {a: v / prob_count for a, v in prob_totals.items()} if prob_count else None

    attn_totals: Dict[str, float] = {}
    attn_count = 0
    for p in predictions:
        attn = getattr(p, "feature_attention", None)
        if isinstance(attn, dict):
            attn_count += 1
            for feat, val in attn.items():
                attn_totals[feat] = attn_totals.get(feat, 0.0) + val
    fused_attn = {f: v / attn_count for f, v in attn_totals.items()} if attn_count else None

    return EnsembleAuthentication(
        trust_score=mean(trust_scores) if trust_scores else None,
        risk_score=mean(risk_scores) if risk_scores else None,
        confidence=mean(confidences) if confidences else None,
        confidence_std=confidence_std,
        decision_probabilities=fused_probs,
        feature_attention=fused_attn,
        # Deliberately left None: the engine derives this via argmax over
        # `decision_probabilities` rather than trusting any one member's
        # discrete recommendation.
        recommended_action=None,
    )
