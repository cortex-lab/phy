# -*- coding: utf-8 -*-

"""Tests of CCG functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ..ccg import (_increment,
                   _diff_shifted,
                   correlograms,
                   )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _random_data(max_cluster):
    sr = 20000
    nspikes = 10000
    spike_samples = np.cumsum(np.random.exponential(scale=.025, size=nspikes))
    spike_samples = (spike_samples * sr).astype(np.uint64)
    spike_clusters = np.random.randint(0, max_cluster, nspikes)
    return spike_samples, spike_clusters


def _ccg_params():
    return .001, .05


def test_utils():
    # First, test _increment().

    # Original array.
    arr = np.arange(10)
    # Indices of elements to increment.
    indices = [0, 2, 4, 2, 2, 2, 2, 2, 2]

    ae(_increment(arr, indices), [1, 1, 9, 3, 5, 5, 6, 7, 8, 9])

    # Then, test _shitdiff.
    # Original array.
    arr = [2, 3, 5, 7, 11, 13, 17]
    # Shifted once.
    ds1 = [1, 2, 2, 4, 2, 4]
    # Shifted twice.
    ds2 = [3, 4, 6, 6, 6]

    ae(_diff_shifted(arr, 1), ds1)
    ae(_diff_shifted(arr, 2), ds2)


def test_ccg_0():
    spike_samples = [0, 10, 10, 20]
    spike_clusters = [0, 1, 0, 1]
    binsize = 1
    winsize_bins = 2 * 3 + 1

    c_expected = np.zeros((2, 2, 4))

    # WARNING: correlograms() is sensitive to the order of identical spike
    # times. This needs to be taken into account when post-processing the
    # CCGs.
    c_expected[1, 0, 0] = 1
    c_expected[0, 1, 0] = 0  # This is a peculiarity of the algorithm.

    c = correlograms(spike_samples, spike_clusters,
                     bin_size=binsize, window_size=winsize_bins,
                     cluster_ids=[0, 1], symmetrize=False)

    ae(c, c_expected)


def test_ccg_1():
    spike_samples = np.array([2, 3, 10, 12, 20, 24, 30, 40], dtype=np.uint64)
    spike_clusters = [0, 1, 0, 0, 2, 1, 0, 2]
    binsize = 1
    winsize_bins = 2 * 3 + 1

    c_expected = np.zeros((3, 3, 4))
    c_expected[0, 1, 1] = 1
    c_expected[0, 0, 2] = 1

    c = correlograms(spike_samples, spike_clusters,
                     bin_size=binsize, window_size=winsize_bins,
                     symmetrize=False)

    ae(c, c_expected)


def test_ccg_2():
    max_cluster = 10
    spike_samples, spike_clusters = _random_data(max_cluster)
    binsize, winsize_bins = _ccg_params()

    c = correlograms(spike_samples, spike_clusters,
                     bin_size=binsize, window_size=winsize_bins,
                     sample_rate=20000, symmetrize=False)

    assert c.shape == (max_cluster, max_cluster, 26)


def test_ccg_symmetry_time():
    """Reverse time and check that the CCGs are just transposed."""

    spike_samples, spike_clusters = _random_data(2)
    binsize, winsize_bins = _ccg_params()

    c0 = correlograms(spike_samples, spike_clusters,
                      bin_size=binsize, window_size=winsize_bins,
                      sample_rate=20000, symmetrize=False)

    spike_samples_1 = np.cumsum(np.r_[np.arange(1),
                                      np.diff(spike_samples)[::-1]])
    spike_samples_1 = spike_samples_1.astype(np.uint64)
    spike_clusters_1 = spike_clusters[::-1]
    c1 = correlograms(spike_samples_1, spike_clusters_1,
                      bin_size=binsize, window_size=winsize_bins,
                      sample_rate=20000, symmetrize=False)

    # The ACGs are identical.
    ae(c0[0, 0], c1[0, 0])
    ae(c0[1, 1], c1[1, 1])

    # The CCGs are just transposed.
    ae(c0[0, 1], c1[1, 0])
    ae(c0[1, 0], c1[0, 1])


def test_ccg_symmetry_clusters():
    """Exchange clusters and check that the CCGs are just transposed."""

    spike_samples, spike_clusters = _random_data(2)
    binsize, winsize_bins = _ccg_params()

    c0 = correlograms(spike_samples, spike_clusters,
                      bin_size=binsize, window_size=winsize_bins,
                      sample_rate=20000, symmetrize=False)

    spike_clusters_1 = 1 - spike_clusters
    c1 = correlograms(spike_samples, spike_clusters_1,
                      bin_size=binsize, window_size=winsize_bins,
                      sample_rate=20000, symmetrize=False)

    # The ACGs are identical.
    ae(c0[0, 0], c1[1, 1])
    ae(c0[1, 1], c1[0, 0])

    # The CCGs are just transposed.
    ae(c0[0, 1], c1[1, 0])
    ae(c0[1, 0], c1[0, 1])


def test_symmetrize_correlograms():
    spike_samples, spike_clusters = _random_data(3)
    binsize, winsize_bins = _ccg_params()

    sym = correlograms(spike_samples, spike_clusters,
                       bin_size=binsize, window_size=winsize_bins,
                       sample_rate=20000)
    assert sym.shape == (3, 3, 51)

    # The ACG are reversed.
    for i in range(3):
        ae(sym[i, i, :], sym[i, i, ::-1])

    # Check that ACG peak is 0.
    assert np.all(sym[np.arange(3), np.arange(3), 25] == 0)

    ae(sym[0, 1, :], sym[1, 0, ::-1])
