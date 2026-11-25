"""Exact diagnostic refractory contamination, independent of display binning.

The fixed contract follows phytastic reference 992b2588. Streaming keeps complete
close components across input boundaries; resource exhaustion never approximates.
"""

import numpy as np

from .live_summaries import _check_cancel

VERSION = 1
BIN_SECONDS = 1800.0


class RefractoryUnavailable(ValueError):  # noqa: N818
    """Required scientific evidence or exact-computation resources are unavailable."""


def deduplicate(ticks, amplitudes, radius, cancel=None):
    """Amplitude-priority suppression in O(n log n), stable on finite ties.

    Scores use float64 absolute raw amplitudes, as in the reference. Signed raw
    values must remain available separately for membership predicates.
    """
    t = np.asarray(ticks)
    if np.asarray(amplitudes).dtype.kind not in 'fiu':
        raise RefractoryUnavailable('Real numeric source amplitudes required.')
    scores = np.abs(np.asarray(amplitudes, dtype=np.float64))
    if not np.isfinite(scores).all():
        raise RefractoryUnavailable('Nonfinite source amplitudes.')
    keep = np.ones(len(t), dtype=bool)
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > radius) + 1, len(t)]
    starts, ends = boundaries[:-1], boundaries[1:]
    pairs = starts[ends - starts == 2]
    keep[np.where(scores[pairs + 1] > scores[pairs], pairs, pairs + 1)] = False
    large = ends - starts > 2
    for start, end in zip(starts[large], ends[large]):
        _check_cancel(cancel)
        local = t[start:end]
        blocked = np.zeros(end - start, dtype=bool)
        chosen = np.zeros(end - start, dtype=bool)
        for k, index in enumerate(np.argsort(-scores[start:end], kind='stable')):
            if k % 4096 == 0:
                _check_cancel(cancel)
            if blocked[index]:
                continue
            chosen[index] = True
            lo = np.searchsorted(local, local[index] - radius)
            hi = np.searchsorted(local, local[index] + radius, side='right')
            blocked[lo:hi] = True
        keep[start:end] = chosen
    return keep


def window_curve(violations, spikes, recorded_seconds, fs):
    """Fixed centered six-hour windows every hour, including the padded endpoint."""
    n_bins = len(spikes)
    centres = np.arange(0, n_bins + 1, 2)
    lo, hi = np.clip(centres - 6, 0, n_bins), np.clip(centres + 6, 0, n_bins)
    cv, cn, cr = (np.r_[0, np.cumsum(x)] for x in (violations, spikes, recorded_seconds))
    v, n, total = cv[hi] - cv[lo], cn[hi] - cn[lo], (cr[hi] - cr[lo]) * fs
    tr, tc = round(1.0 * fs * 1e-3), round(0.3 * fs * 1e-3)
    with np.errstate(invalid='ignore', divide='ignore'):
        arg = 1 - v * (total - 2 * n * tc) / (n.astype(float) ** 2 * (tr - tc))
        contamination = np.where(arg < 0, 1.0, 1 - np.sqrt(np.maximum(arg, 0)))
    return centres / 2, np.where(n >= 1000, contamination, np.nan)


class RefractoryAccumulator:
    """Bounded streaming state for one globally monotonic current membership."""

    def __init__(self, fs, num_samples, valid_intervals, max_component=1000000):
        self.fs, self.num_samples = float(fs), int(num_samples)
        self.radius = round(0.3 * self.fs * 1e-3)
        self.tr = round(1.0 * self.fs * 1e-3)
        self.bin_samples = round(BIN_SECONDS * self.fs)
        if not np.isfinite(fs) or self.radius < 0 or self.tr <= self.radius:
            raise RefractoryUnavailable('Unsupported sample rate.')
        if num_samples <= 0 or self.bin_samples <= 0:
            raise RefractoryUnavailable('Recording duration unavailable.')
        self.n_bins = (self.num_samples + self.bin_samples - 1) // self.bin_samples
        if self.num_samples > np.iinfo(np.int64).max - self.tr or self.n_bins > 200000:
            raise RefractoryUnavailable(
                'Scientific grid or sample arithmetic exceeds resource budget.'
            )
        self.spikes = np.zeros(self.n_bins, dtype=np.int64)
        self.violations = np.zeros(self.n_bins, dtype=np.int64)
        self.recorded = np.zeros(self.n_bins)
        previous = 0
        for a, b in valid_intervals:
            if not (0 <= previous <= a < b <= num_samples):
                raise RefractoryUnavailable('Invalid recorded exposure intervals.')
            for k in range(
                int(a // self.bin_samples),
                min((int(b) + self.bin_samples - 1) // self.bin_samples, self.n_bins),
            ):
                self.recorded[k] += max(
                    0, min(b, (k + 1) * self.bin_samples) - max(a, k * self.bin_samples)
                )
            previous = b
        if not np.any(self.recorded):
            raise RefractoryUnavailable('Recorded exposure unavailable.')
        self.recorded /= self.fs
        self.max_component = int(max_component)
        self.ticks = np.empty(0, dtype=np.int64)
        self.scores = np.empty(0)
        self.partners = np.empty(0, dtype=np.int64)
        self.removed = self.largest_component = 0
        self.last_tick = -1

    def feed(self, ticks, amplitudes, cancel=None, final=False):
        ticks, amplitudes = np.asarray(ticks), np.asarray(amplitudes)
        if amplitudes.dtype.kind not in 'fiu':
            raise RefractoryUnavailable('Real numeric source amplitudes required.')
        if ticks.dtype.kind not in 'iu' or ticks.ndim != 1 or ticks.shape != amplitudes.shape:
            raise RefractoryUnavailable('Integer ticks and scalar raw amplitudes required.')
        if len(ticks) and (ticks.min() < 0 or ticks.max() > np.iinfo(np.int64).max):
            raise RefractoryUnavailable('Source ticks outside signed int64 range.')
        ticks = ticks.astype(np.int64, copy=False)
        if len(ticks):
            if ticks[0] < self.last_tick or np.any(np.diff(ticks) < 0):
                raise RefractoryUnavailable('Source timestamps must be globally monotonic.')
            if ticks[0] < 0 or ticks[-1] >= self.num_samples:
                raise RefractoryUnavailable('Source timestamps outside recording.')
            self.last_tick = int(ticks[-1])
        if not np.isfinite(amplitudes).all():
            raise RefractoryUnavailable('Nonfinite source amplitudes.')
        t = np.concatenate((self.ticks, ticks))
        a = np.concatenate((self.scores, amplitudes))
        gaps = np.diff(t)
        if not np.any(gaps <= self.radius):
            # Isolated spikes cannot suppress one another. Keep the trailing
            # spike pending until the next batch proves its component complete.
            largest = int(bool(len(t)))
            if largest > self.max_component:
                raise RefractoryUnavailable('Exact duplicate component exceeds memory budget.')
            stop = len(t) if final else max(0, len(t) - 1)
            resolved = t[:stop]
        else:
            boundaries = np.r_[0, np.flatnonzero(gaps > self.radius) + 1, len(t)]
            largest = int(np.max(np.diff(boundaries), initial=0))
            if largest > self.max_component:
                raise RefractoryUnavailable('Exact duplicate component exceeds memory budget.')
            stop = len(t) if final else int(boundaries[-2])
            keep = deduplicate(t[:stop], a[:stop], self.radius, cancel)
            self.removed += int(np.count_nonzero(~keep))
            resolved = t[:stop][keep]
        self.largest_component = max(self.largest_component, largest)
        self.spikes += np.bincount(resolved // self.bin_samples, minlength=self.n_bins)
        self.ticks, self.scores = t[stop:].copy(), a[stop:].copy()
        partners = np.concatenate((self.partners, resolved))
        # The first unresolved original tick bounds every future kept tick.
        watermark = t[stop] if stop < len(t) else (self.num_samples + self.tr if final else -1)
        done = len(partners) if final else np.searchsorted(partners, watermark - self.tr)
        # Deduplication guarantees strictly more than radius between retained
        # ticks. Only spikes with a successor inside tr can contribute a pair.
        candidates = np.flatnonzero(np.diff(partners) <= self.tr)
        candidates = candidates[candidates < done]
        if len(candidates):
            earlier = partners[candidates]
            nv = np.searchsorted(partners, earlier + self.tr, side='right') - candidates - 1
            bins = earlier // self.bin_samples
            starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
            self.violations[bins[starts]] += np.add.reduceat(nv, starts)
        self.partners = partners[done:].copy()

    def finish(self, cancel=None):
        self.feed(np.empty(0, dtype=np.int64), np.empty(0), cancel, final=True)
        centres, curve = window_curve(self.violations, self.spikes, self.recorded, self.fs)
        result = {
            'status': 'ready',
            'duplicates_removed': self.removed,
            'spikes': self.spikes,
            'violations': self.violations,
            'recorded_seconds': self.recorded,
            'window_centres_h': centres,
            'contamination': curve,
            'largest_component': self.largest_component,
        }
        for value in result.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        return result
