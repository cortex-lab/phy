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
