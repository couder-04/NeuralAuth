"""
hooks.py
========

Minimal before/after event hook registry so external systems (logging,
tracing, feature flags, real-time alerting, etc.) can observe or extend
the pipeline without modifying DecisionEngine internals.

    hooks = HookRegistry()
    hooks.on("after_fusion", lambda result: send_to_kafka(result))
    engine = DecisionEngine(hooks=hooks)
"""

from __future__ import annotations

from typing import Callable, Dict, List


class HookRegistry:

    _EVENTS = ("before_fusion", "after_fusion", "before_audit", "after_audit")

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {event: [] for event in self._EVENTS}

    def on(self, event: str, callback: Callable) -> None:
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event!r}. Valid events: {self._EVENTS}")
        self._hooks[event].append(callback)

    def fire(self, event: str, **kwargs) -> None:
        for callback in self._hooks.get(event, []):
            callback(**kwargs)
