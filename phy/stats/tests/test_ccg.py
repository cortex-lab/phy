# -*- coding: utf-8 -*-

"""Tests of CCG functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy import array_equal as ae
from pytest import raises

from ..ccg import _increment, _diff_shifted, correlograms


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

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
    ae(_diff_shifted(_diff_shifted(arr)), ds2)


def test_ccg_1():

    spike_times = [2, 3, 10, 12, 20, 24, 30, 40]
    spike_clusters = [0, 1, 0, 0, 2, 1, 0, 2]
    binsize = 1
    winsize_bins = 3 * 2 + 1

    c_expected = np.zeros((3, 3, 3))
    c_expected[0, 1, 1] = 1
    c_expected[0, 0, 2] = 1

    c = correlograms(spike_times, spike_clusters,
                     binsize=binsize, winsize_bins=winsize_bins)

    ae(c, c_expected)


def test_ccg_2():
    sr = 20000
    nspikes = 10000
    spike_times = np.cumsum(np.random.exponential(scale=.002, size=nspikes))
    spike_times = (spike_times * sr).astype(np.int64)
    max_cluster = 10
    spike_clusters = np.random.randint(0, max_cluster, nspikes)

    # window = 50 ms
    winsize_samples = 2 * (25 * 20) + 1
    # bin = 1 ms
    binsize = 1 * 20
    # 51 bins
    winsize_bins = 2 * ((winsize_samples // 2) // binsize) + 1
    assert winsize_bins % 2 == 1

    c = correlograms(spike_times, spike_clusters,
                     binsize=binsize, winsize_bins=winsize_bins)

    assert c.shape == (max_cluster, max_cluster, 26)
