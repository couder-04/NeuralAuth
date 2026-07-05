"""
models/response.py

Output model for the Transaction Authentication Engine.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class TransactionResponse(BaseModel):
    """
    Final response returned by the Transaction Authentication Engine.
    """

    # =====================================================
    # Overall Status
    # =====================================================

    status: str = Field(
        ...,
        description="Overall execution status",
    )

    action: str = Field(
        ...,
        description="Final action decided by the Decision Engine",
    )

    # =====================================================
    # Decision
    # =====================================================

    transaction_allowed: bool = Field(
        ...,
        description="Whether the transaction is allowed",
    )

    authentication_required: bool = Field(
        ...,
        description="Whether additional authentication is required",
    )

    voice_required: bool = Field(
        ...,
        description="Whether voice verification is required",
    )

    otp_required: bool = Field(
        ...,
        description="Whether OTP verification is required",
    )

    manual_review: bool = Field(
        ...,
        description="Whether manual review is required",
    )

    # =====================================================
    # Explanation
    # =====================================================

    message: str = Field(
        ...,
        description="Human-readable decision summary",
    )

    reason: str = Field(
        ...,
        description="Detailed reason for the decision",
    )

    # =====================================================
    # Audit
    # =====================================================

    audit_log: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete audit information",
    )