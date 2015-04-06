# -*- coding: utf-8 -*-

"""Tests of CCG functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ..ccg import _increment, _diff_shifted, correlograms


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _random_data(max_cluster):
    sr = 20000
    nspikes = 10000
    spike_times = np.cumsum(np.random.exponential(scale=.002, size=nspikes))
    spike_times = (spike_times * sr).astype(np.int64)
    spike_clusters = np.random.randint(0, max_cluster, nspikes)
    return spike_times, spike_clusters


def _ccg_params():
    # window = 50 ms
    winsize_samples = 2 * (25 * 20) + 1
    # bin = 1 ms
    binsize = 1 * 20
    # 51 bins
    winsize_bins = 2 * ((winsize_samples // 2) // binsize) + 1
    assert winsize_bins % 2 == 1

    return binsize, winsize_bins


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


def test_ccg_1():
    spike_times = [2, 3, 10, 12, 20, 24, 30, 40]
    spike_clusters = [0, 1, 0, 0, 2, 1, 0, 2]
    binsize = 1
    winsize_bins = 2 * 3 + 1

    c_expected = np.zeros((3, 3, 4))
    c_expected[0, 1, 1] = 1
    c_expected[0, 0, 2] = 1

    c = correlograms(spike_times, spike_clusters,
                     binsize=binsize, winsize_bins=winsize_bins)

    ae(c, c_expected)


def test_ccg_2():
    max_cluster = 10
    spike_times, spike_clusters = _random_data(max_cluster)
    binsize, winsize_bins = _ccg_params()

    c = correlograms(spike_times, spike_clusters,
                     binsize=binsize, winsize_bins=winsize_bins)

    assert c.shape == (max_cluster, max_cluster, 26)


def test_ccg_symmetry_time():
    """Reverse time and check that the CCGs are just transposed."""

    spike_times, spike_clusters = _random_data(2)
    binsize, winsize_bins = _ccg_params()

    c0 = correlograms(spike_times, spike_clusters,
                      binsize=binsize, winsize_bins=winsize_bins)

    spike_times_1 = np.cumsum(np.r_[np.arange(1), np.diff(spike_times)[::-1]])
    spike_clusters_1 = spike_clusters[::-1]
    c1 = correlograms(spike_times_1, spike_clusters_1,
                      binsize=binsize, winsize_bins=winsize_bins)

    # The ACGs are identical.
    ae(c0[0, 0], c1[0, 0])
    ae(c0[1, 1], c1[1, 1])

    # The CCGs are just transposed.
    ae(c0[0, 1], c1[1, 0])
    ae(c0[1, 0], c1[0, 1])


def test_ccg_symmetry_clusters():
    """Exchange clusters and check that the CCGs are just transposed."""

    spike_times, spike_clusters = _random_data(2)
    binsize, winsize_bins = _ccg_params()

    c0 = correlograms(spike_times, spike_clusters,
                      binsize=binsize, winsize_bins=winsize_bins)

    spike_clusters_1 = 1 - spike_clusters
    c1 = correlograms(spike_times, spike_clusters_1,
                      binsize=binsize, winsize_bins=winsize_bins)

    # The ACGs are identical.
    ae(c0[0, 0], c1[1, 1])
    ae(c0[1, 1], c1[0, 0])

    # The CCGs are just transposed.
    ae(c0[0, 1], c1[1, 0])
    ae(c0[1, 0], c1[0, 1])
