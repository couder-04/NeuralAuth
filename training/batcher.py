"""
batcher.py
==========
Split a DataFrame into batches for iterative LLM verification.

No LLM code here. No merging logic. Pure iteration/splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import pandas as pd


@dataclass
class Batch:
    index: int              # 0-based batch number
    df: "pd.DataFrame"       # rows belonging to this batch (original index preserved)

    def row_ids(self, id_column: Optional[str]) -> List:
        if id_column and id_column in self.df.columns:
            return self.df[id_column].tolist()
        return self.df.index.tolist()

    def __len__(self):
        return len(self.df)


class Batcher:
    """Splits a DataFrame into fixed-size batches with resume/limit support."""

    def __init__(self, df: "pd.DataFrame", batch_size: int = 64,
                 shuffle: bool = False, random_seed: int = 42,
                 resume_from_batch: Optional[int] = None,
                 max_batches: Optional[int] = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.resume_from_batch = resume_from_batch or 0
        self.max_batches = max_batches

        # `.sample()` (shuffle path) already returns a new, independent
        # dataframe. When not shuffling, Batcher only ever reads from
        # `work_df` (via `.iloc` slicing in __iter__ / `.index`/`.columns`
        # in Batch.row_ids) and never mutates it, so referencing `df`
        # directly avoids a full, unnecessary copy of the dataset on every
        # run for the (default) shuffle=False case.
        if shuffle:
            work_df = df.sample(frac=1.0, random_state=random_seed)
        else:
            work_df = df

        self._df = work_df
        self._total_batches = max(1, -(-len(work_df) // batch_size))  # ceil div

    @property
    def total_batches(self) -> int:
        return self._total_batches

    @property
    def total_rows(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[Batch]:
        start_batch = self.resume_from_batch
        end_batch = self._total_batches
        if self.max_batches is not None:
            end_batch = min(end_batch, start_batch + self.max_batches)

        for batch_idx in range(start_batch, end_batch):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            chunk = self._df.iloc[start:end]
            if chunk.empty:
                continue
            yield Batch(index=batch_idx, df=chunk)

    def batches_remaining(self) -> int:
        end_batch = self._total_batches
        if self.max_batches is not None:
            end_batch = min(end_batch, self.resume_from_batch + self.max_batches)
        return max(0, end_batch - self.resume_from_batch)
