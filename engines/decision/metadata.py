"""
metadata.py
===========

Builds request-level metadata: request id, decision trace id, model /
policy versions, timestamp, and transaction identifiers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class MetadataBuilder:

    def __init__(self, model_version: str, policy_version: str):
        self.model_version = model_version
        self.policy_version = policy_version

    def build(
        self,
        transaction: Optional[Dict[str, Any]],
        request_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id or str(uuid.uuid4()),
            "decision_trace_id": str(uuid.uuid4()),
            "model_version": self.model_version,
            "policy_version": self.policy_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": (transaction or {}).get("device_id"),
            "transaction_id": (transaction or {}).get("transaction_id"),
        }
