"""
examples.py
===========
Worked calibration examples for the label-verification prompt. Kept
separate from prompts.py so examples can be edited/added without
touching prompt-assembly or domain-spec logic.
"""

WORKED_EXAMPLES = """\
A) WRONG, flag it:
   spoof_probability=0.95, liveness_score=0.04, fraud_history=high -> decision=0 (ALLOW), trust_score=0.88
   Fix: near-certain spoofed/non-live audio + fraud history contradicts high trust and ALLOW.
   Correct trust_score down and decision to 2 or 3. Do not touch spoof_probability/liveness_score/fraud_history.

B) WRONG, flag it:
   failed_attempts=12, fraud_history=high, transaction_risk=0.9 -> risk_score=0.10, decision=0 (ALLOW)
   Fix: risk_score far too low for this evidence. Correct risk_score up, decision to a stricter tier.

C) WRONG, flag it (contradiction in the other direction -- don't only look for fraud):
   speaker_similarity=0.98, liveness_score=0.95, spoof_probability=0.02, stress_score=0.05 -> trust_score=0.15
   Fix: near-ideal identity evidence contradicts a low trust_score. Correct trust_score up.
   This is the same bug class as (A), just inverted -- check both directions on every row.

D) CORRECT as-is, no action:
   previous_trust_score=0.1, failed_attempts=9, spoof_probability=0.8 -> risk_score=0.85, decision=3 (REJECT), trust_score=0.12
   Labels already match the evidence. Return an empty corrections list.

E) NOT a correction (too small to matter):
   trust_score=0.74 when 0.78 would also be reasonable, decision unaffected either way.
   Do not "correct" this -- the current value is within a plausible range. Only flag material
   inconsistencies (label points the wrong direction, or the decision is wrong), not minor
   disagreements where several values would all be defensible.

F) NEVER do this, regardless of how wrong the value looks:
   {"field": "previous_trust_score", ...} / {"field": "transaction_risk", ...} / {"field": "llm_confidence", ...}
   -- these are features, not labels. A correction with any of these as `field` is always invalid.
"""