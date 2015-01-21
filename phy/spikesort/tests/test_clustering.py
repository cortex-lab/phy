# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ...datasets.mock import artificial_spike_clusters
from ..clustering import Clustering, History


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_history():
    history = History()
    assert history.current_item is None

    def _assert_current(item):
        assert id(history.current_item) == id(item)

    item0 = np.zeros(3)
    item1 = np.ones(4)
    item2 = 2 * np.ones(5)

    history.add(item0)
    _assert_current(item0)

    history.add(item1)
    _assert_current(item1)

    assert history.back() is not None
    _assert_current(item0)

    assert history.forward() is not None
    _assert_current(item1)

    assert history.forward() is None
    _assert_current(item1)

    assert history.back() is not None
    _assert_current(item0)
    assert history.back() is None
    assert history.back() is None
    assert len(history) == 3

    history.add(item2)
    assert len(history) == 2
    _assert_current(item2)
    assert history.forward() is None
    assert history.back() is None
    assert history.back() is None


def test_iter_history():
    history = History()

    # Wrong arguments to iter().
    assert len([_ for _ in history.iter(0, 0)]) == 0
    assert len([_ for _ in history.iter(2, 1)]) == 0

    item0 = np.zeros(3)
    item1 = np.ones(4)
    item2 = 2 * np.ones(5)

    history.add(item0)
    history.add(item1)
    history.add(item2)

    for i, item in enumerate(history):
        # Assert item<i>
        if i > 0:
            assert id(item) == id(locals()['item{0:d}'.format(i - 1)])

    for i, item in enumerate(history.iter(1, 2)):
        assert i == 0
        # Assert item<i>
        assert history.current_position == 3
        assert id(item) == id(locals()['item{0:d}'.format(i)])

    for i, item in enumerate(history.iter(2, 3)):
        assert i == 0
        # Assert item<i>
        assert history.current_position == 3
        assert id(item) == id(locals()['item{0:d}'.format(i + 1)])


def test_clustering():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_clusters_base = spike_clusters.copy()

    # Instanciate a Clustering instance.
    clustering = Clustering(spike_clusters)
    assert_array_equal(clustering.spike_clusters, spike_clusters)

    # Test cluster labels.
    assert_array_equal(clustering.cluster_labels, np.arange(n_clusters))

    assert clustering.new_cluster_label() == n_clusters
    assert clustering.n_clusters == n_clusters

    assert clustering.cluster_counts.shape[0] == n_clusters
    assert clustering.cluster_counts.sum() == n_spikes

    # Updating a cluster, method 1.
    spike_clusters_new = spike_clusters.copy()
    spike_clusters_new[:10] = 100
    # This automatically updates the Clustering instance.
    clustering.spike_clusters = spike_clusters_new
    assert_array_equal(clustering.cluster_labels,
                       np.r_[np.arange(n_clusters), 100])

    # Updating a cluster, method 2.
    clustering.spike_clusters = spike_clusters_base
    clustering.spike_clusters[:10] = 100
    # No automatic update (yet?).
    assert_array_equal(clustering.cluster_labels,
                       np.arange(n_clusters))
    # Need to update manually.
    clustering.update()
    assert_array_equal(clustering.cluster_labels,
                       np.r_[np.arange(n_clusters), 100])

    # Assign.
    clustering.assign(slice(None, 10, None), 1000)
    assert 1000 in clustering.cluster_labels
    assert clustering.cluster_counts[-1] == 10
    assert np.all(clustering.spike_clusters[:10] == 1000)

    # Merge.
    count = clustering.cluster_counts.copy()
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [2, 3]))[0]
    my_spikes = clustering.merge([2, 3])
    assert_array_equal(my_spikes, my_spikes_0)
    assert 1001 in clustering.cluster_labels
    assert clustering.cluster_counts[-1] == count[2] + count[3]
    assert np.all(clustering.spike_clusters[my_spikes] == 1001)

    # Merge to a given cluster.
    clustering.spike_clusters = spike_clusters_base
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [4, 6]))[0]
    count = clustering.cluster_counts.copy()
    my_spikes = clustering.merge([4, 6], 11)
    assert_array_equal(my_spikes, my_spikes_0)
    assert 11 in clustering.cluster_labels
    assert clustering.cluster_counts[-1] == count[4] + count[6]
    assert np.all(clustering.spike_clusters[my_spikes] == 11)

    # Split
    my_spikes = [1, 3, 5]
    clustering.split(my_spikes)
    assert np.all(clustering.spike_clusters[my_spikes] == 12)

    clusters = [20, 30, 40]
    clustering.split(my_spikes, clusters)
    assert np.all(clustering.spike_clusters[my_spikes] == clusters)

    # Not implemented (yet) features.
    with raises(NotImplementedError):
        clustering.cluster_labels = np.arange(n_clusters)


def test_clustering_actions():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    clustering = Clustering(spike_clusters)

    checkpoints = {}

    def _checkpoint():
        index = len(checkpoints)
        checkpoints[index] = clustering.spike_clusters.copy()

    def _assert_is_checkpoint(index):
        assert_array_equal(clustering.spike_clusters, checkpoints[index])

    # Checkpoint 0.
    _checkpoint()
    _assert_is_checkpoint(0)

    # Checkpoint 1.
    clustering.merge([0, 1], 11)
    _checkpoint()
    _assert_is_checkpoint(1)

    # Checkpoint 2.
    clustering.merge([2, 3], 12)
    _checkpoint()
    _assert_is_checkpoint(2)

    # Undo once.
    clustering.undo()
    _assert_is_checkpoint(1)

    # Redo.
    clustering.redo()
    _assert_is_checkpoint(2)

    # No redo.
    clustering.redo()
    _assert_is_checkpoint(2)
