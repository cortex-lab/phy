# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ....ext.six import itervalues
from ....io.mock.artificial import artificial_spike_clusters
from ..clustering import (_extend_spikes,
                          _concatenate_spike_clusters,
                          _extend_assignement,
                          Clustering)
from .._update_info import UpdateInfo
from .._utils import _unique, _spikes_in_clusters


#------------------------------------------------------------------------------
# Test assignements
#------------------------------------------------------------------------------

def test_extend_spikes_simple():
    spike_clusters = np.array([3, 5, 2, 9, 5, 5, 2])
    spike_ids = np.array([2, 4, 0])

    # These spikes belong to the following clusters.
    clusters = np.unique(spike_clusters[spike_ids])
    ae(clusters, [2, 3, 5])

    # These are the spikes belonging to those clusters, but not in the
    # originally-specified spikes.
    extended = _extend_spikes(spike_clusters, spike_ids)
    ae(extended, [1, 5, 6])


def test_extend_spikes():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    spike_ids = np.unique(np.random.randint(size=5, low=0, high=n_spikes))

    # These spikes belong to the following clusters.
    clusters = np.unique(spike_clusters[spike_ids])

    # These are the spikes belonging to those clusters, but not in the
    # originally-specified spikes.
    extended = _extend_spikes(spike_clusters, spike_ids)
    assert np.all(np.in1d(spike_clusters[extended], clusters))

    # The function only returns spikes that weren't in the passed spikes.
    assert len(np.intersect1d(extended, spike_ids)) == 0

    # Check that all spikes from our clusters have been selected.
    rest = np.setdiff1d(np.arange(n_spikes), extended)
    rest = np.setdiff1d(rest, spike_ids)
    assert not np.any(np.in1d(spike_clusters[rest], clusters))


def test_concatenate_spike_clusters():
    spikes, clusters = _concatenate_spike_clusters(([1, 5, 4],
                                                    [10, 50, 40]),
                                                   ([2, 0, 3, 6],
                                                    [20, 0, 30, 60]))
    ae(spikes, np.arange(7))
    ae(clusters, np.arange(0, 60 + 1, 10))


def test_extend_assignement():
    spike_clusters = np.array([3, 5, 2, 9, 5, 5, 2])
    spike_ids = np.array([0, 2])

    # These spikes belong to the following clusters.
    clusters = np.unique(spike_clusters[spike_ids])
    ae(clusters, [2, 3])

    # First case: assigning our two spikes to a new cluster.
    # This should not depend on the index chosen.
    for to in (123, 0, 1, 2, 3):
        clusters_rel = [123] * len(spike_ids)
        new_spike_ids, new_cluster_ids = _extend_assignement(spike_clusters,
                                                             spike_ids,
                                                             clusters_rel)
        ae(new_spike_ids, [0, 2, 6])
        ae(new_cluster_ids, [10, 10, 11])

    # Second case: we assign the spikes to different clusters.
    clusters_rel = [0, 1]
    new_spike_ids, new_cluster_ids = _extend_assignement(spike_clusters,
                                                         spike_ids,
                                                         clusters_rel)
    ae(new_spike_ids, [0, 2, 6])
    ae(new_cluster_ids, [10, 11, 12])


#------------------------------------------------------------------------------
# Test clustering
#------------------------------------------------------------------------------

def test_clustering_1():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_clusters_base = spike_clusters.copy()

    # Instanciate a Clustering instance.
    clustering = Clustering(spike_clusters)
    ae(clustering.spike_clusters, spike_clusters)

    # Test clustering.spikes_in_clusters() function.:
    assert np.all(spike_clusters[clustering.spikes_in_clusters([5])] == 5)

    # Test cluster ids.
    ae(clustering.cluster_ids, np.arange(n_clusters))

    assert clustering.new_cluster_id() == n_clusters
    assert clustering.n_clusters == n_clusters

    assert len(clustering.cluster_counts) == n_clusters
    assert sum(itervalues(clustering.cluster_counts)) == n_spikes

    # TODO: test clustering.spikes_per_cluster

    # Updating a cluster, method 1.
    spike_clusters_new = spike_clusters.copy()
    spike_clusters_new[:10] = 100
    clustering.spike_clusters[:] = spike_clusters_new[:]
    # Need to update explicitely.
    clustering._update_spikes_per_cluster()
    ae(clustering.cluster_ids, np.r_[np.arange(n_clusters), 100])

    # Updating a cluster, method 2.
    clustering.spike_clusters[:] = spike_clusters_base[:]
    clustering.spike_clusters[:10] = 100
    # Need to update manually.
    clustering._update_spikes_per_cluster()
    ae(clustering.cluster_ids, np.r_[np.arange(n_clusters), 100])

    # Assign.
    new_cluster = 101
    clustering.assign(np.arange(0, 10), new_cluster)
    assert new_cluster in clustering.cluster_ids
    assert clustering.cluster_counts[new_cluster] == 10
    assert np.all(clustering.spike_clusters[:10] == new_cluster)

    # Merge.
    count = clustering.cluster_counts.copy()
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [2, 3]))[0]
    info = clustering.merge([2, 3])
    my_spikes = info.spikes
    ae(my_spikes, my_spikes_0)
    assert (new_cluster + 1) in clustering.cluster_ids
    assert clustering.cluster_counts[new_cluster + 1] == count[2] + count[3]
    assert np.all(clustering.spike_clusters[my_spikes] == (new_cluster + 1))

    # Merge to a given cluster.
    clustering.spike_clusters[:] = spike_clusters_base[:]
    clustering._update_spikes_per_cluster()
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [4, 6]))[0]
    count = clustering.cluster_counts
    count4, count6 = count[4], count[6]
    info = clustering.merge([4, 6], 11)
    my_spikes = info.spikes
    ae(my_spikes, my_spikes_0)
    assert 11 in clustering.cluster_ids
    assert clustering.cluster_counts[11] == count4 + count6
    assert np.all(clustering.spike_clusters[my_spikes] == 11)

    # Split
    my_spikes = [1, 3, 5]
    clustering.split(my_spikes)
    assert np.all(clustering.spike_clusters[my_spikes] == 12)

    clusters = [0, 1, 2]
    clustering.assign(my_spikes, clusters)
    clu = clustering.spike_clusters[my_spikes]
    ae(clu - clu[0], clusters)


def test_clustering_merge():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    clustering = Clustering(spike_clusters)

    checkpoints = {}

    def _checkpoint():
        index = len(checkpoints)
        checkpoints[index] = clustering.spike_clusters.copy()

    def _assert_is_checkpoint(index):
        ae(clustering.spike_clusters, checkpoints[index])

    def _assert_spikes(clusters):
        ae(info.spikes, _spikes_in_clusters(spike_clusters, clusters))

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
    assert info.added == [7, 8]
    assert info.deleted == [14]
    assert info.count_changed == []
    _assert_is_checkpoint(3)

    # We merge again.
    assert clustering.new_cluster_id() == 14
    assert any(clustering.spike_clusters == 13)
    assert all(clustering.spike_clusters != 14)
    info = clustering.merge([8, 7], 15)
    _assert_spikes([15])
    assert info.added == [15]
    assert info.deleted == [7, 8]
    assert info.count_changed == []
    # Same as checkpoint with 4, but replace 14 with 15.
    res = checkpoints[4]
    res[res == 14] = 15
    ae(clustering.spike_clusters, res)

    # Undo all.
    for i in range(3, -1, -1):
        info = clustering.undo()
        _assert_is_checkpoint(i)

    _assert_is_checkpoint(0)

    # Redo all.
    for i in range(5):
        _assert_is_checkpoint(i)
        info = clustering.redo()


def test_clustering_assign():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    clustering = Clustering(spike_clusters)

    checkpoints = {}

    def _checkpoint(index=None):
        if index is None:
            index = len(checkpoints)
        checkpoints[index] = clustering.spike_clusters.copy()

    def _assert_is_checkpoint(index):
        ae(clustering.spike_clusters, checkpoints[index])

    def _assert_spikes(spikes):
        ae(info.spikes, spikes)

    # Checkpoint 0.
    _checkpoint()
    _assert_is_checkpoint(0)

    my_spikes_1 = np.unique(np.random.randint(low=0, high=n_spikes, size=5))
    my_spikes_2 = np.unique(np.random.randint(low=0, high=n_spikes, size=10))
    my_spikes_3 = np.unique(np.random.randint(low=0, high=n_spikes, size=1000))
    my_spikes_4 = np.arange(n_spikes - 5)

    # Checkpoint 1.
    info = clustering.split(my_spikes_1)  # Split to 10.
    _checkpoint()
    _assert_spikes(my_spikes_1)
    assert info.added == [10]
    assert info.deleted == []
    assert len(info.count_changed) <= 5
    _assert_is_checkpoint(1)

    # Checkpoint 2.
    info = clustering.split(my_spikes_2)  # Split to 11.
    _checkpoint()
    _assert_spikes(my_spikes_2)
    assert info.added == [11]
    assert info.deleted == []
    assert len(info.count_changed) <= 10
    _assert_is_checkpoint(2)

    # Checkpoint 3.
    info = clustering.assign(my_spikes_3, 20)  # Assign to 20.
    _checkpoint()
    _assert_spikes(my_spikes_3)
    assert info.added == [20]
    assert len(info.count_changed) >= 5
    _assert_is_checkpoint(3)

    # Undo checkpoint 3.
    info = clustering.undo()
    _checkpoint()
    # _assert_spikes(my_spikes_3)
    assert info.deleted == [20]
    assert len(info.count_changed) >= 5
    _assert_is_checkpoint(2)

    # Checkpoint 4.
    info = clustering.assign(my_spikes_4, 30)  # Assign to 30.
    _checkpoint(4)
    _assert_spikes(my_spikes_4)
    assert info.added == [30]
    assert len(info.deleted) >= 2
    assert len(info.count_changed) >= 2
    _assert_is_checkpoint(4)
