"""Bounded lookup and scheduling shared by offline streaming passes."""

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor

import numpy as np


def check_cancel(cancel):
    if cancel is not None and cancel():
        raise CancelledError()


def worker_count(workers):
    """Reject accidental unbounded or silently truncated worker requests."""
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('workers must be an integer between 1 and 32.')
    return workers


def bounded_map(function, items, workers=1):
    """Yield ordered results with at most workers tasks and results resident."""
    worker_count(workers)
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is None:
                    break
                pending.append(pool.submit(function, item))
            while pending:
                yield pending.popleft().result()
                item = next(iterator, None)
                if item is not None:
                    pending.append(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()


class UnitRows:
    """Validated unit-to-row map, with a capped dense fast path for compact IDs."""

    def __init__(self, units, max_bytes=8 * 1024 * 1024):
        self.units = np.asarray(units)
        if (
            self.units.ndim != 1
            or self.units.dtype.kind not in 'iu'
            or np.any(self.units < 0)
            or np.any(self.units[1:] <= self.units[:-1])
        ):
            raise ValueError('Unit catalog must contain sorted unique nonnegative integers.')
        self.table = None
        # Avoid a huge allocation for a handful of sparse, high-valued IDs.
        limit = min(max_bytes, max(4096, len(self.units) * 64))
        size = int(self.units[-1]) + 1 if len(self.units) else 0
        if size * 4 <= limit and len(self.units) <= np.iinfo(np.int32).max:
            self.table = np.full(size, -1, dtype=np.int32)
            self.table[self.units] = np.arange(len(self.units), dtype=np.int32)

    def __call__(self, values):
        values = np.asarray(values)
        if values.dtype.kind not in 'iu' or np.any(values < 0):
            raise ValueError('Source unit catalog mismatch.')
        if self.table is not None:
            if np.any(values >= len(self.table)):
                raise ValueError('Source unit catalog mismatch.')
            rows = self.table[values]
            if np.any(rows < 0):
                raise ValueError('Source unit catalog mismatch.')
        else:
            rows = np.searchsorted(self.units, values)
            if np.any(rows >= len(self.units)) or np.any(self.units[rows] != values):
                raise ValueError('Source unit catalog mismatch.')
        return rows
