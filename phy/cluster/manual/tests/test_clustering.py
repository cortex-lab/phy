# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ....ext.six import itervalues
from ....datasets.mock import artificial_spike_clusters
from ..clustering import (_count_clusters, _diff_counts,
                          Clustering)
from .._update_info import UpdateInfo
from .._utils import _unique, _spikes_in_clusters


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_counts():
    c1 = {1: 10, 2: 20, 3: 30, 5: 50}
    c2 = {0: 5, 2: 20, 3: 31}
    ui = _diff_counts(c1, c2)
    assert ui.added == [0]
    assert ui.deleted == [1, 5]
    assert ui.count_changed == [3]


def test_update_info():

    # Check default values in UpdateInfo.
    info = UpdateInfo(deleted=[1, 2])
    assert info.added == []
    assert info.deleted == [1, 2]

    n_spikes = 1000
    n_clusters = 10

    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    cluster_counts_before = _count_clusters(spike_clusters)

    assert len(cluster_counts_before) == 10
    assert sum(itervalues(cluster_counts_before)) == 1000
    assert cluster_counts_before[5] > 0

    assert cluster_counts_before[100] == 0

    # Change the first 10 spikes and assign them to cluster 100.
    spike_labels = np.arange(10)
    cluster_labels = 100 * np.ones(10, dtype=np.int32)

    spike_clusters[spike_labels] = cluster_labels
    cluster_counts_after = _count_clusters(spike_clusters)

    info = _diff_counts(cluster_counts_before, cluster_counts_after)
    assert info.added == [100]
    assert info.deleted == []
    assert len(info.count_changed) > 0


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

    assert len(clustering.cluster_counts) == n_clusters
    assert sum(itervalues(clustering.cluster_counts)) == n_spikes

    # Updating a cluster, method 1.
    spike_clusters_new = spike_clusters.copy()
    spike_clusters_new[:10] = 100
    clustering.spike_clusters[:] = spike_clusters_new[:]
    # Need to update explicitely.
    clustering.update_cluster_counts()
    assert_array_equal(clustering.cluster_labels,
                       np.r_[np.arange(n_clusters), 100])

    # Updating a cluster, method 2.
    clustering.spike_clusters[:] = spike_clusters_base[:]
    clustering.spike_clusters[:10] = 100
    # Need to update manually.
    clustering.update_cluster_counts()
    assert_array_equal(clustering.cluster_labels,
                       np.r_[np.arange(n_clusters), 100])

    # Assign.
    clustering.assign(slice(None, 10, None), 1000)
    assert 1000 in clustering.cluster_labels
    assert clustering.cluster_counts[1000] == 10
    assert np.all(clustering.spike_clusters[:10] == 1000)

    # Merge.
    count = clustering.cluster_counts.copy()
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [2, 3]))[0]
    info = clustering.merge([2, 3])
    my_spikes = info.spikes
    assert_array_equal(my_spikes, my_spikes_0)
    assert 1001 in clustering.cluster_labels
    assert clustering.cluster_counts[1001] == count[2] + count[3]
    assert np.all(clustering.spike_clusters[my_spikes] == 1001)

    # Merge to a given cluster.
    clustering.spike_clusters[:] = spike_clusters_base[:]
    clustering.update_cluster_counts()
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [4, 6]))[0]
    count = clustering.cluster_counts
    count4, count6 = count[4], count[6]
    info = clustering.merge([4, 6], 11)
    my_spikes = info.spikes
    assert_array_equal(my_spikes, my_spikes_0)
    assert 11 in clustering.cluster_labels
    assert clustering.cluster_counts[11] == count4 + count6
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

    def _assert_spikes(clusters):
        assert_array_equal(info.spikes,
                           _spikes_in_clusters(spike_clusters, clusters))

    # Checkpoint 0.
    _checkpoint()
    _assert_is_checkpoint(0)

    # Checkpoint 1.
    info = clustering.merge([0, 1], 11)
    _checkpoint()
    _assert_spikes([11])
    assert info.added == [11]
    assert info.deleted == [0, 1]
    assert info.count_changed == []
    _assert_is_checkpoint(1)

    # Checkpoint 2.
    info = clustering.merge([2, 3], 12)
    _checkpoint()
    _assert_spikes([12])
    assert info.added == [12]
    assert info.deleted == [2, 3]
    assert info.count_changed == []
    _assert_is_checkpoint(2)

    # Undo once.
    info = clustering.undo()
    _assert_spikes([2, 3])
    assert info.added == [2, 3]
    assert info.deleted == [12]
    assert info.count_changed == []
    _assert_is_checkpoint(1)

    # Redo.
    info = clustering.redo()
    _assert_spikes([12])
    assert info.added == [12]
    assert info.deleted == [2, 3]
    assert info.count_changed == []
    _assert_is_checkpoint(2)

    # No redo.
    info = clustering.redo()
    _assert_is_checkpoint(2)

    # Merge again.
    info = clustering.merge([4, 5, 6], 13)
    _checkpoint()
    _assert_spikes([13])
    assert info.added == [13]
    assert info.deleted == [4, 5, 6]
    assert info.count_changed == []
    _assert_is_checkpoint(3)

    # One more merge.
    info = clustering.merge([8, 7])  # merged to 14
    _checkpoint()
    _assert_spikes([14])
    assert info.added == [14]
    assert info.deleted == [7, 8]
    assert info.count_changed == []
    _assert_is_checkpoint(4)

    # Now we undo.
    info = clustering.undo()
    _assert_is_checkpoint(3)

    # We merge again.
    assert clustering.new_cluster_label() == 14
    assert any(clustering.spike_clusters == 13)
    assert all(clustering.spike_clusters != 14)
    info = clustering.merge([8, 7], 15)
    # Same as checkpoint with 4, but replace 14 with 15.
    res = checkpoints[4]
    res[res == 14] = 15
    assert_array_equal(clustering.spike_clusters, res)

    # Undo all.
    for i in range(3, -1, -1):
        info = clustering.undo()
        _assert_is_checkpoint(i)

    _assert_is_checkpoint(0)

    # Redo all.
    for i in range(5):
        _assert_is_checkpoint(i)
        info = clustering.redo()
