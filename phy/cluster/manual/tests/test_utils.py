# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from .._utils import _unique, _spikes_in_clusters
from ....datasets.mock import artificial_spike_clusters


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_utils():
    """Test clustering utility functions."""
    _unique([])

    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    assert_array_equal(_unique(spike_clusters), np.arange(n_clusters))

    assert_array_equal(_spikes_in_clusters(spike_clusters, []), [])

    for i in range(10):
        assert np.all(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                         [1])] == 1)

    clusters = [1, 5, 9]
    assert np.all(np.in1d(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                             clusters)],
                          clusters))
