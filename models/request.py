"""
models/request.py

Input request model for the Transaction Authentication Engine.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    Input request to the Transaction Authentication Engine.
    """

    # =====================================================
    # User
    # =====================================================

    user_id: str = Field(..., description="Unique user ID")

    # =====================================================
    # Voice Input
    # =====================================================

    transcript: str = Field(
        ...,
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

    gps_latitude: Optional[float] = None

    gps_longitude: Optional[float] = None

    vehicle_speed: float = 0.0

    engine_running: bool = True

    timestamp: Optional[str] = None

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