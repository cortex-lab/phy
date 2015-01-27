# -*- coding: utf-8 -*-

"""Tests of CCG functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy import array_equal as ae
from pytest import raises

from ..ccg import _increment, _shiftdiff, Correlograms


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_utils():
    a = np.arange(10)
    b = [0, 2, 4, 2, 2, 2, 2, 2, 2]
    print(_increment(a, b))

    print(_shiftdiff(np.array([2, 3, 5, 7, 11, 13, 17]), 1))
    print(_shiftdiff(np.array([2, 3, 5, 7, 11, 13, 17]), 2))


def test_ccg():
    sr = 20000
    nspikes = 10000
    spiketimes = np.cumsum(np.random.exponential(scale=.002, size=nspikes))
    spiketimes = (spiketimes * sr).astype(np.int64)
    maxcluster = 10
    spike_clusters = np.random.randint(0, maxcluster, nspikes)

    winsize_samples = 2*(25 * 20) + 1
    binsize = 1 * 20  # 1 ms
    winsize_bins = 2 * ((winsize_samples // 2) // binsize) + 1
    assert winsize_bins % 2 == 1

    c = Correlograms(spiketimes, binsize, winsize_bins)
    correlograms = c.compute(spike_clusters, [])
    assert correlograms is not None
