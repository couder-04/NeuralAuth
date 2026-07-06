"""
history.py
==========

Decision history moved OUT of DecisionEngine into a pluggable
HistoryStore interface.

Why: an in-process history (e.g. a deque on `self`) is only correct for
a single process. Behind Gunicorn/uvicorn with multiple workers — or
multiple machines — each process would keep its own, inconsistent
history. In production, implement this interface against Redis,
Postgres, Kafka, etc., and inject it into DecisionEngine:

    engine = DecisionEngine(history_store=RedisHistoryStore(redis_client))

InMemoryHistoryStore is provided for local development and tests only.
NullHistoryStore is provided for deployments where decision history /
fraud analytics is handled entirely by a separate system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List


class HistoryStore(ABC):

    @abstractmethod
    def append(self, key: str, action: str) -> None:
        ...

    @abstractmethod
    def recent(self, key: str, limit: int = 10) -> List[str]:
        ...


class InMemoryHistoryStore(HistoryStore):
    """Process-local history store. Fine for local dev/tests; NOT safe
    across multiple workers or machines in production."""

    def __init__(self, maxlen: int = 10):
        self._maxlen = maxlen
        self._store: Dict[str, Deque[str]] = {}

    def append(self, key: str, action: str) -> None:
        self._store.setdefault(key, deque(maxlen=self._maxlen)).append(action)

    def recent(self, key: str, limit: int = 10) -> List[str]:
        return list(self._store.get(key, deque()))[-limit:]


class NullHistoryStore(HistoryStore):
    """No-op store for deployments where history is tracked elsewhere."""

    def append(self, key: str, action: str) -> None:
        return

    def recent(self, key: str, limit: int = 10) -> List[str]:
        return []
