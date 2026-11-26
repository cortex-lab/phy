"""Synthetic lookup/histogram microbenchmark; not end-to-end preparation."""

import time
import tracemalloc

import numpy as np

from phy.apps.large_recording.streaming import UnitRows

rng = np.random.default_rng(0)
n = 10_000_000
units = np.arange(1173, dtype=np.int64)
bins = 2050
batch = 100000
ids = rng.integers(0, len(units), n, dtype=np.int32)
cols = np.arange(n, dtype=np.int64) * bins // n
values = rng.standard_normal(n).astype(np.float32)
lookup = UnitRows(units)


def run(new):
    counts = np.zeros((len(units), bins), np.int64)
    sums = np.zeros_like(counts, dtype=float)
    finite_counts = np.zeros_like(counts)
    tracemalloc.start()
    start = time.perf_counter()
    for first in range(0, n, batch):
        v = values[first : first + batch]
        c = cols[first : first + batch]
        r = (
            lookup(ids[first : first + batch])
            if new
            else np.searchsorted(units, ids[first : first + batch])
        )
        if new:
            lo, hi = int(c[0]), int(c[-1]) + 1
            width = hi - lo
            flat = r.astype(np.int64) * width + c - lo
            for out, weights in ((counts, None), (sums, v), (finite_counts, None)):
                out[:, lo:hi] += np.bincount(
                    flat, weights=weights, minlength=len(units) * width
                ).reshape(-1, width)
        else:
            np.add.at(counts, (r, c), 1)
            np.add.at(sums, (r, c), v)
            np.add.at(finite_counts, (r, c), 1)
    seconds = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (counts, sums, finite_counts), seconds, peak


old, t0, m0 = run(False)
new, t1, m1 = run(True)
print('spikes', n, 'units', len(units), 'bins', bins, 'batch', batch)
print(
    'baseline seconds',
    t0,
    'scratch peak bytes',
    m0,
    'new seconds',
    t1,
    'scratch peak bytes',
    m1,
    'speedup',
    t0 / t1,
)
print('counts exact', np.array_equal(old[0], new[0]), 'sums exact', np.array_equal(old[1], new[1]))
for dtype in (np.int32, np.int64):
    keys = ids[:batch].astype(dtype)
    start = time.perf_counter()
    for _ in range(100):
        np.argsort(keys, kind='stable')
    print('stable grouping', dtype, 'seconds/100', time.perf_counter() - start)
