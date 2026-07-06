"""
models/request.py

Input request model for the Transaction Authentication Engine.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """
    Input request to the Transaction Authentication Engine.
    """

    # =====================================================
    # User
    # =====================================================

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique user ID",
    )

    # =====================================================
    # Voice Input
    # =====================================================

    transcript: str = Field(
        ...,
        min_length=1,
        description="Speech transcript produced by ASR",
    )

    audio_path: Optional[str] = Field(
        default=None,
        description="Optional path to recorded audio",
    )

    # =====================================================
    # Vehicle
    # =====================================================

    vehicle_id: Optional[str] = None

    gps_latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Decimal degrees, WGS84 (-90..90)",
    )

    gps_longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Decimal degrees, WGS84 (-180..180)",
    )

    vehicle_speed: float = Field(
        default=0.0,
        ge=0.0,
        description="Speed in km/h; must be non-negative",
    )

    engine_running: bool = True

    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of the transaction event",
    )

    # =====================================================
    # Session
    # =====================================================

    session_id: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # =====================================================
    # Feature Extraction Input
    # =====================================================
    #
    # These are passed directly to FeatureExtractor.extract()
    #

    identity: Dict[str, Any] = Field(default_factory=dict)

    biometric: Dict[str, Any] = Field(default_factory=dict)

    behavior: Dict[str, Any] = Field(default_factory=dict)

    vehicle: Dict[str, Any] = Field(default_factory=dict)

    history: Dict[str, Any] = Field(default_factory=dict)

    transaction: Dict[str, Any] = Field(default_factory=dict)

    # =====================================================
    # Field validation
    # =====================================================

    @field_validator("user_id", "transcript")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """`min_length=1` alone still accepts whitespace-only strings
        (e.g. `" "`); reject those too since a blank user_id/transcript
        is never meaningful input."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("timestamp")
    @classmethod
    def _timestamp_must_be_iso8601(cls, value: Optional[str]) -> Optional[str]:
        """Reject clearly malformed timestamps at the API boundary
        (422) rather than letting them silently degrade to "no
        wall-clock signal" deep in the Policy Engine (see
        engines/policy_context.parse_request_timestamp, which is still
        kept lenient for internal/programmatic callers -- the API
        boundary is where we can and should give the caller a clear
        error instead)."""
        if value is None:
            return value
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "timestamp must be an ISO-8601 datetime string"
            ) from exc
        return value