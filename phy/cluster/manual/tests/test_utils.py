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
                      _flatten_spikes_per_cluster,
                      _concatenate_per_cluster_arrays)
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


def test_concatenate_per_cluster_arrays():
    """Test _spikes_per_cluster()."""

    def _column(arr):
        out = np.zeros((len(arr), 10))
        out[:, 0] = arr
        return out

    # 8, 11, 12, 13, 17, 18, 20
    spikes_per_cluster = {2: [11, 13, 17], 3: [8, 12], 5: [18, 20]}

    arrays_1d = {2: [1, 3, 7], 3: [8, 2], 5: [8, 0]}

    arrays_2d = {2: _column([1, 3, 7]),
                 3: _column([8, 2]),
                 5: _column([8, 0])}

    concat = _concatenate_per_cluster_arrays(spikes_per_cluster, arrays_1d)
    ae(concat, [8, 1, 2, 3, 7, 8, 0])

    concat = _concatenate_per_cluster_arrays(spikes_per_cluster, arrays_2d)
    ae(concat[:, 0], [8, 1, 2, 3, 7, 8, 0])
    ae(concat[:, 1:], np.zeros((7, 9)))
