"""Independent all-pairs and discrete-clock oracles for visual CCG exclusion."""

from itertools import product

import numpy as np
import pytest

from phy.apps.large_recording.statistics import _window_chance_baseline, window_correlograms

TOGGLES = list(product((False, True), repeat=2))


def _all_pairs(samples, clusters, windows, ids, bin_samples, half_bins, radius, auto, cross):
    """Enumerate unordered source rows directly, independently of the shift kernel."""
    expected = np.zeros((len(ids), len(ids), 2 * half_bins + 1), dtype=np.int64)
    lookup = {label: i for i, label in enumerate(ids)}
    for start, stop in windows:
        rows = [i for i, tick in enumerate(samples) if start <= tick < stop and clusters[i] in ids]
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                lag = int(samples[right]) - int(samples[left])
                a, b = lookup[clusters[left]], lookup[clusters[right]]
                if lag <= radius and (auto if a == b else cross):
                    continue
                delay_bin = lag // bin_samples
                if delay_bin > half_bins:
                    continue
                expected[a, b, half_bins + delay_bin] += 1
                if a != b or delay_bin:
                    expected[b, a, half_bins - delay_bin] += 1
    return expected


@pytest.mark.parametrize('auto,cross', TOGGLES)
@pytest.mark.parametrize(
    'width,radius', [(0, 0), (0.1, 3), (0.3, 9), (0.4, 12), (0.32, 9), (0.5, 15), (10, 300)]
)
def test_exclusion_matches_independent_pairs_at_late_clocks(width, radius, auto, cross):
    origin = 1000 * 3600 * 30000
    offsets = np.array([0, 0, 1, 3, 9, 10, 12, 14, 15, 16, 23, 30, 44, 45, 60])
    samples = origin + np.concatenate((offsets, [99, 100], 300 + offsets))
    clusters = np.resize(np.array([2_000_000_001, 2, 2, 2_000_000_001, 99]), len(samples))
    windows = np.array([[origin, origin + 100], [origin + 300, origin + 400]])
    original_samples, original_clusters = samples.copy(), clusters.copy()
    # Swap tied rows and reverse output cluster order independently. Unselected
    # cluster 99 and a selected spike in the gap do not contribute.
    for swap_ties, ids in product((False, True), ([2_000_000_001, 2], [2, 2_000_000_001])):
        labels = clusters.copy()
        if swap_ties:
            labels[[0, 1]] = labels[[1, 0]]
        expected = _all_pairs(samples, labels, windows, ids, 15, 3, radius, auto, cross)
        actual = window_correlograms(
            samples,
            labels,
            windows,
            ids,
            30000,
            0.0005,
            0.003,
            exclusion_ms=width,
            exclude_auto=auto,
            exclude_cross=cross,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, actual.transpose(1, 0, 2)[..., ::-1])
        assert actual.dtype == np.int64
    np.testing.assert_array_equal(samples, original_samples)
    np.testing.assert_array_equal(clusters, original_clusters)


@pytest.mark.parametrize('auto,cross', TOGGLES)
def test_empty_and_fully_excluded_windows(auto, cross):
    samples = np.array([0, 0, 1, 2], dtype=np.int64)
    clusters = np.array([3, 9, 3, 9])
    windows = np.array([[0, 3], [100, 110]])
    hist, baseline = window_correlograms(
        samples,
        clusters,
        windows,
        [9, 3, 42],
        1000,
        0.001,
        0.004,
        True,
        exclusion_ms=100,
        exclude_auto=auto,
        exclude_cross=cross,
    )
    expected = _all_pairs(samples, clusters, windows, [9, 3, 42], 1, 2, 100, auto, cross)
    np.testing.assert_array_equal(hist, expected)
    np.testing.assert_array_equal(hist[2], 0)
    np.testing.assert_array_equal(baseline[2], 0)
    if auto:
        np.testing.assert_array_equal(hist[np.arange(3), np.arange(3)], 0)
    if cross:
        np.testing.assert_array_equal(hist[0, 1], 0)


def test_cutoff_is_on_pair_lag_before_binning_and_does_not_delete_spikes():
    # 0.1 and 0.4 ms occupy the same central 0.5 ms bin at 30 kHz.
    # All three spikes remain eligible as partners at longer delays.
    samples = np.array([0, 3, 12, 60], dtype=np.int64)
    clusters = np.array([0, 1, 1, 0])
    hist = window_correlograms(
        samples,
        clusters,
        [[0, 100]],
        [0, 1],
        30000,
        0.0005,
        0.004,
        exclude_cross=True,
    )
    assert hist[0, 1, 4] == 1  # 0.4 ms survives; 0.1 ms does not.
    assert hist[0, 1, 1] == 2  # Both former near-zero partners pair with the later spike.
    assert hist[1, 1, 4] == 1  # Auto pairs remain enabled independently.


@pytest.mark.parametrize('width', [-1, -0.01, np.nan, np.inf, -np.inf, None, 'invalid'])
def test_exclusion_width_must_be_finite_and_nonnegative_even_when_disabled(width):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        window_correlograms(
            np.array([0]), np.array([0]), [[0, 1]], [0], 30000, 0.0005, 0.004, exclusion_ms=width
        )


def test_legacy_positional_baseline_and_disabled_options_preserve_output():
    samples = np.array([0, 0, 3, 12, 60])
    clusters = np.array([0, 1, 1, 0, 1])
    args = samples, clusters, [[0, 100]], [1, 0], 30000, 0.0005, 0.004, True
    expected = window_correlograms(*args)
    actual = window_correlograms(*args, exclusion_ms=100, exclude_auto=False, exclude_cross=False)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(TypeError):
        window_correlograms(*args, 0.3)


def _clock_baseline(counts, length, bin_samples, half_bins, radius, auto, cross):
    """Sum over every ordered pair of independent uniform clock positions."""
    expected = np.zeros((len(counts), len(counts)), dtype=float)
    if not half_bins:
        return expected
    for a, n_a in enumerate(counts):
        for b, n_b in enumerate(counts):
            allowed = 0
            for left, right in product(range(length), repeat=2):
                lag = right - left
                if not 1 <= lag // bin_samples <= half_bins:
                    continue
                if lag <= radius and (auto if a == b else cross):
                    continue
                allowed += 1
            partners = n_a * (n_b - (a == b))
            expected[a, b] = partners * allowed / length**2 / half_bins
    return expected


@pytest.mark.parametrize('auto,cross', TOGGLES)
def test_chance_baseline_exhaustive_uniform_clock_oracle(auto, cross):
    counts = np.array([3, 2, 1, 0])
    for length, bin_samples, half_bins, radius in product(
        range(1, 8), (1, 2, 3), (0, 1, 3), range(10)
    ):
        actual = _window_chance_baseline(
            counts,
            length,
            bin_samples,
            half_bins,
            exclusion_samples=radius,
            exclude_auto=auto,
            exclude_cross=cross,
        )
        expected = _clock_baseline(counts, length, bin_samples, half_bins, radius, auto, cross)
        np.testing.assert_allclose(actual, expected, atol=1e-15)


@pytest.mark.parametrize('auto,cross', TOGGLES)
@pytest.mark.parametrize('width,radius', [(0, 0), (0.3, 0), (2, 2), (3.5, 3), (100, 100)])
def test_returned_baseline_matches_window_local_oracle(auto, cross, width, radius):
    samples = np.array([0, 1, 1, 2, 100, 101, 102])
    clusters = np.array([0, 0, 1, 1, 0, 1, 1])
    windows = np.array([[0, 7], [100, 104]])
    _, actual = window_correlograms(
        samples,
        clusters,
        windows,
        [1, 0, 9],
        1000,
        0.002,
        0.012,
        True,
        exclusion_ms=width,
        exclude_auto=auto,
        exclude_cross=cross,
    )
    expected = np.zeros((3, 3))
    for start, stop in windows:
        counts = [
            np.sum((samples >= start) & (samples < stop) & (clusters == c)) for c in [1, 0, 9]
        ]
        expected += _clock_baseline(counts, stop - start, 2, 3, radius, auto, cross)
    np.testing.assert_allclose(actual, expected)


def test_default_cutoff_with_half_ms_bins_leaves_off_center_baseline_unchanged():
    samples = np.array([0, 3, 12, 30, 60])
    clusters = np.array([0, 0, 1, 1, 0])
    args = samples, clusters, [[0, 100]], [0, 1], 30000, 0.0005, 0.004, True
    _, expected = window_correlograms(*args)
    _, actual = window_correlograms(*args, exclude_auto=True, exclude_cross=True)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('auto,cross', TOGGLES)
def test_window_accumulation_and_central_fold_preserve_int64(monkeypatch, auto, cross):
    # Isolate accumulation from pair generation: each per-window count fits in
    # int32, while the sum and folded cross-unit center exceed its maximum.
    from phy.apps.large_recording import statistics

    def positive(*args, **kwargs):
        return np.full((2, 2, 2), 2**30, dtype=np.int64)

    monkeypatch.setattr(statistics, 'correlograms', positive)
    monkeypatch.setattr(statistics, '_excluded_positive_correlograms', positive)
    hist = window_correlograms(
        np.array([0, 10, 20]),
        np.array([0, 1, 0]),
        [[0, 2], [10, 12], [20, 22]],
        [0, 1],
        1000,
        0.001,
        0.002,
        exclude_auto=auto,
        exclude_cross=cross,
    )
    assert hist.dtype == np.int64
    assert hist[0, 0, 1] == 3 * 2**30
    assert hist[0, 1, 1] == 6 * 2**30
    assert hist[0, 1, 0] == 3 * 2**30
