"""Descriptive statistics restricted to retained contiguous observation windows."""

from decimal import ROUND_FLOOR, Decimal

import numpy as np
from phylib.stats import correlograms


def window_isi(samples, windows, sample_rate):
    intervals = []
    for start, end in windows:
        first, last = np.searchsorted(samples, [start, end])
        intervals.append(np.diff(samples[first:last]) / sample_rate)
    return np.concatenate(intervals) if intervals else np.array([], dtype=np.float64)


def _window_chance_baseline(
    counts,
    n_samples,
    bin_samples,
    n_positive_bins,
    *,
    exclusion_samples=0,
    exclude_auto=False,
    exclude_cross=False,
):
    """Mean off-center bin expectation under uniform independent sample clocks.

    Condition on each unit's count in this observation window, and place its
    spikes independently with replacement on the ``n_samples`` integer clocks.
    At a positive delay d there are ``n_samples - d`` possible clock pairs.
    Average their probabilities over the displayed positive bins, including
    zero exposure where a bin extends beyond this window. Negative bins have
    the same expectation. Auto pairs use the factorial count n*(n-1), excluding
    self-pairs. The folded central bin has a different convention and is excluded.
    Optional pair exclusion also removes its clock exposure, while the mean's
    denominator remains the number of all displayed off-center bins.

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
    probability = exposure / length**2 / n_positive_bins
    if not (exclude_auto or exclude_cross):
        return pairs * probability
    # A cutoff can remove part of an off-center bin. Keep the denominator equal
    # to the number of displayed off-center bins, including fully masked bins.
    first = max(first, exclusion_samples + 1)
    n_delays = max(0, last - first + 1)
    exposure = n_delays * (2 * length - first - last) / 2
    filtered = exposure / length**2 / n_positive_bins
    probabilities = np.full(pairs.shape, filtered if exclude_cross else probability)
    np.fill_diagonal(probabilities, filtered if exclude_auto else probability)
    return pairs * probabilities


def _exclusion_sample_radius(exclusion_ms, sample_rate):
    """Floor a decimal millisecond radius without rounding an integer boundary down."""
    try:
        width = float(exclusion_ms)
    except (TypeError, ValueError) as error:
        raise ValueError('exclusion_ms must be finite and nonnegative.') from error
    if not np.isfinite(width) or width < 0:
        raise ValueError('exclusion_ms must be finite and nonnegative.')
    # Decimal input spellings preserve boundaries such as 0.3 ms * 30000 Hz = 9,
    # while genuinely fractional radii still exclude only integer delays below it.
    radius = Decimal(str(exclusion_ms)) * Decimal(str(sample_rate)) / 1000
    return min(int(radius.to_integral_value(rounding=ROUND_FLOOR)), 2**63 - 1)


def _excluded_positive_correlograms(
    ticks,
    labels,
    n_clusters,
    sample_rate,
    bin_size,
    window_size,
    exclusion_samples,
    exclude_auto,
    exclude_cross,
):
    """Count integer-lag pairs with a temporary exclusion mask before binning.

    Like phylib's shift kernel, use linear temporary storage and retire a source
    only when its partners exceed the maximum displayed lag. A short excluded
    pair must not retire that source: its later partners may have valid delays.
    """
    assert sample_rate > 0
    assert np.all(np.diff(ticks) >= 0), 'The spike times must be increasing.'
    bin_size = np.clip(bin_size, 1e-5, 1e5)
    bin_samples = int(sample_rate * bin_size)
    assert bin_samples >= 1
    window_size = np.clip(window_size, 1e-5, 1e5)
    n_positive_bins = int(0.5 * window_size / bin_size)
    result = np.zeros((n_clusters, n_clusters, n_positive_bins + 1), dtype=np.int64)
    active = np.ones(len(ticks), dtype=bool)
    shift = 1
    while active[:-shift].any():
        delays = ticks[shift:] - ticks[:-shift]
        bins = delays // bin_samples
        active[:-shift][bins > n_positive_bins] = False
        keep = active[:-shift].copy()
        if exclude_auto and exclude_cross:
            keep &= delays > exclusion_samples
        else:
            same = labels[:-shift] == labels[shift:]
            eligible = same if exclude_auto else ~same
            keep &= ~eligible | (delays > exclusion_samples)
        indices = np.ravel_multi_index(
            (labels[:-shift][keep], labels[shift:][keep], bins[keep]), result.shape
        )
        increments = np.bincount(indices)
        result.ravel()[: len(increments)] += increments
        shift += 1
    return result


def window_correlograms(
    samples,
    clusters,
    windows,
    cluster_ids,
    sample_rate,
    bin_size,
    window_size,
    return_baseline=False,
    *,
    exclusion_ms=0.3,
    exclude_auto=False,
    exclude_cross=False,
):
    """Exact window-local pair counts and an optional observed-window chance level.

    Positive/negative lag bins follow phylib's floor-to-bin convention. The
    central bin counts unordered within-unit pairs and cross-unit pairs with
    absolute delays below one bin. Cross-unit central counts sum both temporal
    orientations, making them independent of equal-time source row order.

    Optional visual pair exclusion removes integer sample delays whose absolute
    value is at most ``exclusion_ms`` milliseconds before binning, independently
    for same-unit (``exclude_auto``) and cross-unit (``exclude_cross``) pairs.
    Both switches default to False. Zero width removes exact ties when enabled.
    Selected spikes and observation windows remain unchanged.

    With ``return_baseline=True``, return ``(counts, chance)``. The horizontal
    ``chance`` matrix sums the per-window conditional-uniform mean off-center
    expectations described in ``_window_chance_baseline``. It uses precisely the
    same selected spikes, integer bin width and finite observed windows as the
    histogram; gaps and full-recording counts do not contribute. It is not the
    expected central-bin height and is zero when there are no off-center bins.
    """
    exclusion_samples = _exclusion_sample_radius(exclusion_ms, sample_rate)
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
        # Avoid phylib's lookup allocation indexed by absolute cluster ID.
        labels = order[np.searchsorted(sorted_ids, clusters[first:last][mask])]
        counts = np.bincount(labels, minlength=len(cluster_ids))
        if np.any(counts > 46340):
            raise ValueError(
                'Evidence window has too many spikes for exact correlogram counts; prepare shorter windows.'
            )
        if exclude_auto or exclude_cross:
            positive = _excluded_positive_correlograms(
                ticks,
                labels,
                len(cluster_ids),
                sample_rate,
                bin_size,
                window_size,
                exclusion_samples,
                exclude_auto,
                exclude_cross,
            )
        else:
            # phylib truncates float times * rate to integer samples. A common
            # quarter-sample offset preserves every integer tick after conversion
            # without changing any pair delay, including late-clock recordings.
            times = (ticks.astype(np.float64) + 0.25) / sample_rate
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
                exclusion_samples=exclusion_samples,
                exclude_auto=exclude_auto,
                exclude_cross=exclude_cross,
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
