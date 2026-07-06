"""
policy_context.py
==================

Bridges the numeric signals the Feature Extractor already computes
(`FeatureVector.location_familiarity`, `FeatureVector.time_familiarity`
-- both 0..1 similarity scores, see engines/feature_extractor.py) into
the categorical labels `PolicyInput`/`rules/policy_rules.yaml` actually
key off of (`"FAMILIAR" | "UNFAMILIAR"`, `"NORMAL" | "ODD_HOUR"`).

Why this exists as its own module instead of being inlined at the
`PolicyInput(...)` call site in api/server.py:

* `api/server.py` was previously hardcoding these fields
  (`location_familiarity="FAMILIAR"`, `time_familiarity="NORMAL"`),
  which silently made the `UnfamiliarLocation` and `OddHourActivity`
  policy rules unreachable no matter what the request actually
  contained. Centralizing the real derivation in one place makes that
  class of bug easier to see and to test in isolation.
* Thresholds/heuristics for "what counts as unfamiliar" are a policy
  decision, not a request-parsing detail -- keeping them here (rather
  than scattered across API call sites) means they can be tuned, or
  swapped for a smarter signal later (e.g. a real geofencing service,
  or a per-user learned hour-of-day distribution), without touching
  `api/server.py` or the Policy Engine's YAML-driven rule contract.

Explicit assumptions (documented here so they're easy to revisit):

1. `location_familiarity` / `time_familiarity` scores are produced
   upstream (currently: taken as-is from the request's `vehicle`
   payload, see `models/request.py` / `engines/feature_extractor.py`)
   as a continuous 0..1 similarity to the user's historical pattern.
   There is, as of this module, no independent geofencing/IP-based
   signal or per-user learned "usual activity hours" model -- when one
   exists, it should be threaded in here as an *additional* signal
   rather than replacing the threshold check below.
2. A transaction can also be flagged `ODD_HOUR` from the request's own
   wall-clock `timestamp` (if supplied), independent of the user's
   personal familiarity score, using a fixed "objectively unusual"
   clock window. This is a conservative default (00:00-05:59) and is
   deliberately configurable per call, since what counts as "odd" is a
   business decision, not a technical one.
3. `timestamp`, if supplied, is assumed to be an ISO-8601 string in a
   fixed offset or naive-local form parseable by
   `datetime.fromisoformat`. If `TransactionRequest` starts carrying
   explicit per-request timezone info, `parse_request_timestamp`
   should be updated to normalize it before extracting `.hour`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults -- tunable in one place, see module docstring assumption (1)/(2).
# ---------------------------------------------------------------------------

DEFAULT_LOCATION_FAMILIARITY_THRESHOLD = 0.5
DEFAULT_TIME_FAMILIARITY_THRESHOLD = 0.5

# Inclusive hour range (24h clock) treated as "objectively unusual"
# regardless of the user's personal familiarity score.
DEFAULT_ODD_HOUR_RANGE: Tuple[int, int] = (0, 5)


def classify_location_familiarity(
    score: float,
    threshold: float = DEFAULT_LOCATION_FAMILIARITY_THRESHOLD,
) -> str:
    """
    Map a continuous 0..1 `FeatureVector.location_familiarity` score
    into the `"FAMILIAR" | "UNFAMILIAR"` label `PolicyInput` and
    `rules/policy_rules.yaml` (`UnfamiliarLocation` rule) expect.
    """
    return "FAMILIAR" if score >= threshold else "UNFAMILIAR"


def parse_request_timestamp(raw_timestamp: Optional[str]) -> Optional[datetime]:
    """
    Best-effort parse of `TransactionRequest.timestamp` into a
    `datetime`. Returns `None` (rather than raising) if the field is
    absent or not a recognizable ISO-8601 string -- a missing/invalid
    timestamp degrades to "no wall-clock signal available", not a hard
    failure, since the familiarity-score signal is still usable on its
    own (see `classify_time_familiarity`).
    """
    if not raw_timestamp:
        return None
    try:
        return datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return None


def classify_time_familiarity(
    score: float,
    timestamp: Optional[datetime] = None,
    threshold: float = DEFAULT_TIME_FAMILIARITY_THRESHOLD,
    odd_hour_range: Tuple[int, int] = DEFAULT_ODD_HOUR_RANGE,
) -> str:
    """
    Map a continuous 0..1 `FeatureVector.time_familiarity` score (and,
    optionally, the request's own wall-clock hour) into the
    `"NORMAL" | "ODD_HOUR"` label `PolicyInput` and
    `rules/policy_rules.yaml` (`OddHourActivity` rule) expect.

    Two independent signals can each mark a transaction `ODD_HOUR`:
      1. `score < threshold`     -- unusual *for this user*.
      2. `timestamp` falls inside `odd_hour_range` -- unusual by a
         fixed, user-independent clock window (only checked if a
         timestamp was actually parsed; see assumption (2) above).
    """
    if score < threshold:
        return "ODD_HOUR"

    if timestamp is not None:
        lo, hi = odd_hour_range
        if lo <= timestamp.hour <= hi:
            return "ODD_HOUR"

    return "NORMAL"
