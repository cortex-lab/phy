# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from .._utils import (_unique, _spikes_in_clusters, _spikes_per_cluster,
                      _flatten_spikes_per_cluster)
from ....io.mock.artificial import artificial_spike_clusters


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_unique():
    """Test _unique()."""

    _unique([])

    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    # Test _unique().
    ae(_unique(spike_clusters), np.arange(n_clusters))


def test_spikes_in_clusters():
    """Test _spikes_in_clusters()."""

    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    ae(_spikes_in_clusters(spike_clusters, []), [])

    for i in range(n_clusters):
        assert np.all(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                         [i])] == i)

    clusters = [1, 5, 9]
    assert np.all(np.in1d(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                             clusters)],
                          clusters))


def test_spikes_per_cluster():
    """Test _spikes_per_cluster()."""

    n_spikes = 1000
    spike_ids = np.arange(n_spikes).astype(np.int64)
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    spikes_per_cluster = _spikes_per_cluster(spike_ids, spike_clusters)
    assert list(spikes_per_cluster.keys()) == list(range(n_clusters))

    for i in range(10):
        ae(spikes_per_cluster[i], np.sort(spikes_per_cluster[i]))
        assert np.all(spike_clusters[spikes_per_cluster[i]] == i)

    sc = _flatten_spikes_per_cluster(spikes_per_cluster)
    ae(spike_clusters, sc)
