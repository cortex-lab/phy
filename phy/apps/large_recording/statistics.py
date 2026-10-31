"""Descriptive statistics restricted to retained contiguous observation windows."""

import numpy as np
from phylib.stats import correlograms


def window_isi(samples, windows, sample_rate):
    intervals = []
    for start, end in windows:
        first, last = np.searchsorted(samples, [start, end])
        intervals.append(np.diff(samples[first:last]) / sample_rate)
    return np.concatenate(intervals) if intervals else np.array([], dtype=np.float64)


def window_correlograms(
    samples, clusters, windows, cluster_ids, sample_rate, bin_size, window_size
):
    """Exact window-local pair counts, without a full-duration rate baseline.

    Positive/negative lag bins follow phylib's floor-to-bin convention. The
    central bin counts unordered within-unit pairs and cross-unit pairs with
    absolute delays below one bin. Cross-unit central counts sum both temporal
    orientations, making them independent of equal-time source row order.
    """
    selected = np.isin(clusters, cluster_ids)
    order = np.argsort(cluster_ids)
    sorted_ids = np.asarray(cluster_ids)[order]
    result = None
    for start, end in windows:
        first, last = np.searchsorted(samples, [start, end])
        mask = selected[first:last]
        ticks = samples[first:last][mask] - start
        if len(ticks) and ticks.max() >= 2**50:
            raise ValueError(
                'Observation window is too long for exact correlogram sample conversion.'
            )
        # phylib truncates float times * rate to integer samples. A common
        # quarter-sample offset preserves every integer tick after conversion
        # without changing any pair delay, including late-clock recordings.
        times = (ticks.astype(np.float64) + 0.25) / sample_rate
        # Avoid phylib's lookup allocation indexed by absolute cluster ID.
        labels = order[np.searchsorted(sorted_ids, clusters[first:last][mask])]
        if np.any(np.bincount(labels, minlength=len(cluster_ids)) > 46340):
            raise ValueError(
                'Evidence window has too many spikes for exact correlogram counts; prepare shorter windows.'
            )
        positive = correlograms(
            times,
            labels,
            sample_rate=sample_rate,
            cluster_ids=np.arange(len(cluster_ids)),
            bin_size=bin_size,
            window_size=window_size,
            symmetrize=False,
        ).astype(np.int64)
        # Accumulate before symmetrization to preserve exact central counts and
        # int64 totals across many windows. Self-spike pairs are never counted.
        if result is None:
            result = positive
        else:
            result += positive
    if result is None:
        raise ValueError('Correlograms require at least one observation window.')
    center = result[..., 0] + result[..., 0].T
    np.fill_diagonal(center, np.diag(result[..., 0]))
    result[..., 0] = center
    return np.dstack((result[..., 1:][..., ::-1].transpose(1, 0, 2), result))
