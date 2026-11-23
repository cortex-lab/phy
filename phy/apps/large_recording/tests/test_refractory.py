"""Independent pairwise oracle, boundary streaming, and scientific grid tests."""

from concurrent.futures import CancelledError

import numpy as np
import pytest

from phy.apps.large_recording.refractory import (
    RefractoryAccumulator,
    RefractoryUnavailable,
    deduplicate,
    window_curve,
)


def brute(t, a, radius=9):
    chosen = []
    for i in sorted(range(len(t)), key=lambda i: (-abs(float(a[i])), i)):
        if all(abs(int(t[i]) - int(t[j])) > radius for j in chosen):
            chosen.append(i)
    return np.array(sorted(chosen), dtype=int)


@pytest.mark.parametrize('seed', range(12))
def test_independent_oracle_and_every_stream_boundary(seed):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(0, 16, 220)).astype(np.int64)
    a = rng.integers(-4, 5, len(t)).astype(np.float32)
    keep = brute(t, a)
    np.testing.assert_array_equal(np.flatnonzero(deduplicate(t, a, 9)), keep)
    kept = t[keep]
    expected_v = sum(9 < int(y) - int(x) <= 30 for i, x in enumerate(kept) for y in kept[i + 1 :])
    for chunk in (1, 2, 9, 17, 1000):
        acc = RefractoryAccumulator(30000, 100000, [[0, 1000], [1100, 100000]])
        for start in range(0, len(t), chunk):
            acc.feed(t[start : start + chunk], a[start : start + chunk])
        r = acc.finish()
        assert r['duplicates_removed'] == len(t) - len(kept)
        assert r['spikes'].sum() == len(kept)
        assert r['violations'].sum() == expected_v
        assert r['recorded_seconds'].sum() == pytest.approx(99900 / 30000)


def test_bin_boundary_later_partner_gap_and_padded_endpoint():
    edge = 1800 * 30000
    t = np.array([edge - 20, edge + 10, edge + 40], dtype=np.int64)
    acc = RefractoryAccumulator(30000, edge * 3 + 1, [[0, edge], [edge + 5, edge * 3 + 1]])
    acc.feed(t, np.ones(3))
    result = acc.finish()
    np.testing.assert_array_equal(result['spikes'], [1, 2, 0, 0])
    np.testing.assert_array_equal(result['violations'], [1, 1, 0, 0])
    np.testing.assert_array_equal(result['window_centres_h'], [0, 1, 2])
    assert result['window_centres_h'][-1] > acc.num_samples / acc.fs / 3600
    assert np.isnan(result['contamination']).all()


def test_cutoff_saturation_no_extra_clipping():
    _, curve = window_curve(np.array([100]), np.array([1000]), np.array([1800.0]), 30000)
    np.testing.assert_array_equal(curve, [1.0])
    _, curve = window_curve(np.array([100]), np.array([999]), np.array([1800.0]), 30000)
    assert np.isnan(curve).all()


def test_nonfinite_chain_budget_cancel_and_original_signed_precision():
    t = np.arange(200, dtype=np.int64)
    acc = RefractoryAccumulator(30000, 1000, [[0, 1000]], max_component=100)
    with pytest.raises(RefractoryUnavailable, match='budget'):
        acc.feed(t, np.ones(200))
    with pytest.raises(RefractoryUnavailable, match='Nonfinite'):
        deduplicate(t[:2], [1, np.nan], 9)
    with pytest.raises(RefractoryUnavailable, match='Real numeric'):
        deduplicate(t[:2], [1 + 2j, 3 + 4j], 9)
    with pytest.raises(CancelledError):
        deduplicate(t, np.ones(200), 9, lambda: True)
    np.testing.assert_array_equal(
        deduplicate(np.array([0, 1]), [-1.0, 1.0 + 1e-12], 9), [False, True]
    )
    np.testing.assert_array_equal(deduplicate(np.array([0, 1]), [-1.0, 1.0], 9), [True, False])


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'uint32', 'uint64'])
def test_integer_tick_normalization_and_unsigned_overflow(dtype):
    acc = RefractoryAccumulator(30000, 1000, [[0, 1000]])
    acc.feed(np.array([0, 9, 30], dtype=dtype), np.array([1.0, 2.0, 3.0]))
    result = acc.finish()
    assert result['spikes'].sum() == 2
    assert result['violations'].sum() == 1
    acc = RefractoryAccumulator(30000, 1000, [[0, 1000]])
    with pytest.raises(RefractoryUnavailable, match='int64'):
        acc.feed(np.array([2**64 - 1], dtype=np.uint64), np.array([1.0]))


@pytest.mark.parametrize('fs,tc,tr', [(5000, 2, 5), (2500, 1, 2), (1500, 0, 2)])
def test_python_sample_rounding_and_inclusive_pair_endpoint(fs, tc, tr):
    acc = RefractoryAccumulator(fs, fs * 10, [[0, fs * 10]])
    assert acc.radius == tc and acc.tr == tr
    acc.feed(np.array([0, tc, tc + tr], dtype=np.int64), np.array([1.0, 2.0, 3.0]))
    result = acc.finish()
    assert result['duplicates_removed'] == 1
    assert result['spikes'].sum() == 2
    assert result['violations'].sum() == 1
