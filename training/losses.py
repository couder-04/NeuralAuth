"""
training/losses.py

Loss factory for the Authentication Network.

This module intentionally stays lightweight. The actual implementation of
the multi-task authentication loss lives inside
engines.authentication_network.UncertaintyWeightedLoss.

Keeping this wrapper separate allows the training pipeline to remain clean
and makes it easy to swap loss functions in the future.
"""

from __future__ import annotations

import torch.nn as nn

from engines.authentication_network import (
    ModelConfig,
    UncertaintyWeightedLoss,
)


def build_loss(config: ModelConfig) -> nn.Module:
    """
    Build the loss function used during training.

    Args:
        config: Model configuration.

    Returns:
        Configured loss module.
    """
    return UncertaintyWeightedLoss(config)


def get_loss_name(config: ModelConfig) -> str:
    """
    Returns a human-readable description of the active loss.
    """

    if config.use_uncertainty_weighting:
        return "UncertaintyWeightedLoss"

    return "WeightedMultiTaskLoss"


__all__ = [
    "build_loss",
    "get_loss_name",
]