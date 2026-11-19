"""Descriptive statistics restricted to retained contiguous observation windows."""

import numpy as np
from phylib.stats import correlograms


def window_isi(samples, windows, sample_rate):
    intervals = []
    for start, end in windows:
        first, last = np.searchsorted(samples, [start, end])
        intervals.append(np.diff(samples[first:last]) / sample_rate)
    return np.concatenate(intervals) if intervals else np.array([], dtype=np.float64)


def _window_chance_baseline(counts, n_samples, bin_samples, n_positive_bins):
    """Mean off-center bin expectation under uniform independent sample clocks.

    Condition on each unit's count in this observation window, and place its
    spikes independently with replacement on the ``n_samples`` integer clocks.
    At a positive delay d there are ``n_samples - d`` possible clock pairs.
    Average their probabilities over the displayed positive bins, including
    zero exposure where a bin extends beyond this window. Negative bins have
    the same expectation. Auto pairs use the factorial count n*(n-1), excluding
    self-pairs. The folded central bin has a different convention and is excluded.

    This is an exact mean of the off-center null expectations, not an exact
    null curve (which tapers with lag), a confidence bound, or a recording-wide
    firing rate. It assumes stationarity only within this evidence window.
    """
    counts = np.asarray(counts, dtype=np.float64)
    pairs = np.outer(counts, counts)
    pairs[np.diag_indices(len(counts))] -= counts
    if not n_positive_bins:
        return np.zeros_like(pairs)
    length = int(n_samples)
    first = int(bin_samples)
    last = min((int(n_positive_bins) + 1) * first - 1, length - 1)
    n_delays = max(0, last - first + 1)
    exposure = n_delays * (2 * length - first - last) / 2
    return pairs * (exposure / length**2 / n_positive_bins)


def window_correlograms(
    samples,
    clusters,
    windows,
    cluster_ids,
    sample_rate,
    bin_size,
    window_size,
    return_baseline=False,
):
    """Exact window-local pair counts and an optional observed-window chance level.

    Positive/negative lag bins follow phylib's floor-to-bin convention. The
    central bin counts unordered within-unit pairs and cross-unit pairs with
    absolute delays below one bin. Cross-unit central counts sum both temporal
    orientations, making them independent of equal-time source row order.

    With ``return_baseline=True``, return ``(counts, chance)``. The horizontal
    ``chance`` matrix sums the per-window conditional-uniform mean off-center
    expectations described in ``_window_chance_baseline``. It uses precisely the
    same selected spikes, integer bin width and finite observed windows as the
    histogram; gaps and full-recording counts do not contribute. It is not the
    expected central-bin height and is zero when there are no off-center bins.
    """
    selected = np.isin(clusters, cluster_ids)
    order = np.argsort(cluster_ids)
    sorted_ids = np.asarray(cluster_ids)[order]
    result = None
    baseline = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.float64)
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
        counts = np.bincount(labels, minlength=len(cluster_ids))
        if np.any(counts > 46340):
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
        if return_baseline:
            baseline += _window_chance_baseline(
                counts,
                end - start,
                int(sample_rate * np.clip(bin_size, 1e-5, 1e5)),
                positive.shape[2] - 1,
            )
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
    result = np.dstack((result[..., 1:][..., ::-1].transpose(1, 0, 2), result))
    return (result, baseline) if return_baseline else result
