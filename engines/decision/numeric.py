"""
numeric.py
==========

Small shared helper so every module can safely convert framework tensors
(e.g. torch.Tensor) into plain Python numbers/lists without each module
re-implementing the same try/except.
"""

from __future__ import annotations

from typing import Any, Optional


def to_python(value: Any) -> Optional[Any]:
    if value is None:
        return None

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    return value
