"""Independent pair-count oracle and observation-boundary regressions."""

import numpy as np

from phy.apps.large_recording.statistics import window_correlograms, window_isi


def test_window_isi_excludes_missing_observation_periods_and_preserves_ties():
    samples = np.array([0, 1, 1, 2, 1000, 1002], dtype=np.int64)
    windows = np.array([[0, 3], [1000, 1003]], dtype=np.int64)
    np.testing.assert_array_equal(window_isi(samples, windows, 1), [1, 0, 1, 2])


def test_late_sample_ccg_matches_all_pairs_oracle_with_sparse_ids():
    origin = 1000 * 3600 * 30000
    samples = origin + np.array([0, 1, 30, 30, 31, 60, 300, 301, 330], dtype=np.int64)
    clusters = np.array([2_000_000_000, 2, 2, 2_000_000_000, 2, 2, 2, 2_000_000_000, 2])
    windows = np.array([[origin, origin + 100], [origin + 300, origin + 400]], dtype=np.int64)
    ids = [2_000_000_000, 2]
    actual = window_correlograms(samples, clusters, windows, ids, 30000, 0.001, 0.004)
    expected = np.zeros((2, 2, 5), dtype=np.int64)
    lookup = {cluster: index for index, cluster in enumerate(ids)}
    for start, end in windows:
        rows = [i for i, tick in enumerate(samples) if start <= tick < end]
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                delay = int(samples[right] - samples[left]) // 30
                if delay > 2:
                    continue
                a, b = lookup[clusters[left]], lookup[clusters[right]]
                expected[a, b, 2 + delay] += 1
                if a != b or delay != 0:
                    expected[b, a, 2 - delay] += 1
    np.testing.assert_array_equal(actual, expected)


def test_cross_unit_central_pairs_do_not_depend_on_tied_row_order():
    samples = np.array([0, 0, 10, 10], dtype=np.int64)
    windows = np.array([[0, 20]], dtype=np.int64)
    a = window_correlograms(samples, np.array([0, 1, 1, 0]), windows, [0, 1], 30000, 0.001, 0.004)
    b = window_correlograms(samples, np.array([1, 0, 1, 0]), windows, [0, 1], 30000, 0.001, 0.004)
    np.testing.assert_array_equal(a, b)
    assert a[0, 1, 2] == 4


def test_window_chance_baseline_matches_discrete_uniform_all_pairs():
    """Enumerate independent clock positions; exclude the folded central bin."""
    from phy.apps.large_recording.statistics import _window_chance_baseline

    for length, bin_samples, half_bins in ((7, 2, 3), (3, 2, 4), (2, 3, 2)):
        counts = np.array([3, 2, 1, 0])
        positive_pairs = sum(
            1
            for left in range(length)
            for right in range(length)
            if 1 <= (right - left) // bin_samples <= half_bins
        )
        pair_probability = positive_pairs / length**2 / half_bins
        expected = np.outer(counts, counts).astype(float)
        expected[np.diag_indices(len(counts))] -= counts
        expected *= pair_probability
        np.testing.assert_allclose(
            _window_chance_baseline(counts, length, bin_samples, half_bins), expected
        )
        assert expected[2, 2] == 0  # One spike has no auto-pair partner.


def test_window_correlograms_returns_baseline_from_per_window_counts():
    samples = np.array([0, 1, 2, 3, 100, 101, 102], dtype=np.int64)
    clusters = np.array([9, 9, 2, 2, 9, 9, 9])
    windows = np.array([[0, 10], [100, 105]], dtype=np.int64)
    hist, baseline = window_correlograms(
        samples, clusters, windows, [9, 2, 99], 1000, 0.002, 0.008, return_baseline=True
    )
    # Off-center bins contain delays [2, 3] and [4, 5]. Count their finite
    # exposure independently: (8+7+6+5)/100/2 and (3+2+1)/25/2.
    first_probability = 26 / 100 / 2
    second_probability = 6 / 25 / 2
    expected = np.array(
        [
            [2 * first_probability + 6 * second_probability, 4 * first_probability, 0],
            [4 * first_probability, 2 * first_probability, 0],
            [0, 0, 0],
        ]
    )
    np.testing.assert_allclose(baseline, expected)
    np.testing.assert_array_equal(
        hist, window_correlograms(samples, clusters, windows, [9, 2, 99], 1000, 0.002, 0.008)
    )


def test_window_chance_baseline_matches_independent_poisson_realizations():
    """Simulate a null independently, including auto factorial and central counts."""
    rng = np.random.default_rng(12341)
    length, trials, half_bins, bin_samples = 40, 1000, 3, 4
    samples, clusters, windows = [], [], []
    expected_center = np.zeros((2, 2), dtype=float)
    for trial in range(trials):
        start = trial * length
        windows.append([start, start + length])
        counts = rng.poisson([5, 8])
        central_probability = (
            sum(
                abs(left - right) < bin_samples
                for left in range(length)
                for right in range(length)
            )
            / length**2
        )
        expected_center[0, 0] += counts[0] * (counts[0] - 1) / 2 * central_probability
        expected_center[1, 1] += counts[1] * (counts[1] - 1) / 2 * central_probability
        expected_center[0, 1] += counts[0] * counts[1] * central_probability
        for label, count in enumerate(counts):
            samples.extend(start + rng.integers(length, size=count))
            clusters.extend([label] * count)
    order = np.argsort(samples)
    hist, baseline = window_correlograms(
        np.asarray(samples)[order],
        np.asarray(clusters)[order],
        np.asarray(windows),
        [0, 1],
        1000,
        bin_samples / 1000,
        2 * half_bins * bin_samples / 1000,
        return_baseline=True,
    )
    off_center = np.delete(hist, half_bins, axis=2).mean(axis=2)
    np.testing.assert_allclose(off_center, baseline, rtol=0.06)
    expected_center[1, 0] = expected_center[0, 1]
    np.testing.assert_allclose(hist[..., half_bins], expected_center, rtol=0.06)
    # Cross central counts fold both temporal orientations into one wider bin.
    assert hist[0, 1, half_bins] > 1.8 * baseline[0, 1]


def test_window_chance_baseline_uses_actual_integer_bins_and_excludes_center_only():
    samples = np.array([0, 1, 2, 3], dtype=np.int64)
    clusters = np.array([0, 0, 1, 1])
    windows = np.array([[0, 10]], dtype=np.int64)
    quantized = window_correlograms(
        samples, clusters, windows, [0, 1], 1000, 0.0029, 0.013, return_baseline=True
    )
    exact = window_correlograms(
        samples, clusters, windows, [0, 1], 1000, 0.002, 0.008, return_baseline=True
    )
    for actual, expected in zip(quantized, exact):
        np.testing.assert_array_equal(actual, expected)
    center_only, baseline = window_correlograms(
        samples, clusters, windows, [0, 1], 1000, 0.002, 0.003, return_baseline=True
    )
    assert center_only.shape == (2, 2, 1)
    assert center_only.max() > 0
    np.testing.assert_array_equal(baseline, np.zeros((2, 2)))
