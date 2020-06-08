# -*- coding: utf-8 -*-

"""Test clustering."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from phylib.io.mock import artificial_spike_clusters
from phylib.io.array import (_spikes_in_clusters,)
from phylib.utils import connect
from ..clustering import (_extend_spikes,
                          _concatenate_spike_clusters,
                          _extend_assignment,
                          Clustering)


#------------------------------------------------------------------------------
# Test assignments
#------------------------------------------------------------------------------

def test_extend_spikes_simple():
    spike_clusters = np.array([3, 5, 2, 9, 5, 5, 2])
    spike_ids = np.array([2, 4, 0])

    # These spikes belong to the following clusters.
    clusters = np.unique(spike_clusters[spike_ids])
    ae(clusters, [2, 3, 5])

    # These are the spikes belonging to those clusters, but not in the
    # originally-specified spikes.
    extended = _extend_spikes(spike_ids, spike_clusters)
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
    extended = _extend_spikes(spike_ids, spike_clusters)
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


def test_extend_assignment():
    spike_clusters = np.array([3, 5, 2, 9, 5, 5, 2])
    spike_ids = np.array([0, 2])

    # These spikes belong to the following clusters.
    clusters = np.unique(spike_clusters[spike_ids])
    ae(clusters, [2, 3])

    # First case: assigning our two spikes to a new cluster.
    # This should not depend on the index chosen.
    for to in (123, 0, 1, 2, 3):
        clusters_rel = [123] * len(spike_ids)
        new_spike_ids, new_cluster_ids = _extend_assignment(spike_ids,
                                                            spike_clusters,
                                                            clusters_rel,
                                                            10,
                                                            )
        ae(new_spike_ids, [0, 2, 6])
        ae(new_cluster_ids, [10, 10, 11])

    # Second case: we assign the spikes to different clusters.
    clusters_rel = [0, 1]
    new_spike_ids, new_cluster_ids = _extend_assignment(spike_ids,
                                                        spike_clusters,
                                                        clusters_rel,
                                                        10,
                                                        )
    ae(new_spike_ids, [0, 2, 6])
    ae(new_cluster_ids, [10, 11, 12])


#------------------------------------------------------------------------------
# Test clustering
#------------------------------------------------------------------------------

def test_clustering_split():
    spike_clusters = np.array([2, 5, 3, 2, 7, 5, 2])

    # Instantiate a Clustering instance.
    clustering = Clustering(spike_clusters)
    ae(clustering.spike_clusters, spike_clusters)
    n_spikes = len(spike_clusters)
    assert clustering.n_spikes == n_spikes
    ae(clustering.spike_ids, np.arange(n_spikes))

    splits = [[0],
              [1],
              [2],
              [0, 1],
              [0, 2],
              [1, 2],
              [0, 1, 2],
              [3],
              [4],
              [3, 4],
              [6],
              [6, 5],
              [0, 6],
              [0, 3, 6],
              [0, 2, 6],
              np.arange(7)]

    # Test many splits.
    for to_split in splits:
        clustering.reset()
        clustering.split(to_split)

    # Test many splits, without reset this time.
    clustering.reset()
    for to_split in splits:
        clustering.split(to_split)


def test_clustering_descendants_merge():
    spike_clusters = np.array([2, 5, 3, 2, 7, 5, 2])

    # Instantiate a Clustering instance.
    clustering = Clustering(spike_clusters)

    # Test merges.
    with raises(ValueError):
        clustering.merge(2, 3)

    up = clustering.merge([2, 3])
    new = up.added[0]
    assert new == 8
    assert set(up.descendants) == set([(2, 8), (3, 8)])

    with raises(ValueError):
        up = clustering.merge([2, 8])

    up = clustering.merge([5, 8])
    new = up.added[0]
    assert new == 9
    assert set(up.descendants) == set([(5, 9), (8, 9)])


def test_clustering_descendants_split():
    spike_clusters = np.array([2, 5, 3, 2, 7, 5, 2])

    # Instantiate a Clustering instance.
    clustering = Clustering(spike_clusters)

    with raises(Exception):
        clustering.split([-1])
    with raises(Exception):
        clustering.split([8])

    # First split.
    up = clustering.split([0])
    assert up.deleted == [2]
    assert up.added == [8, 9]
    assert set(up.descendants) == set([(2, 8), (2, 9)])
    ae(clustering.spike_clusters, [8, 5, 3, 9, 7, 5, 9])

    # Undo.
    up = clustering.undo()
    assert up.deleted == [8, 9]
    assert up.added == [2]
    assert set(up.descendants) == set([(8, 2), (9, 2)])
    ae(clustering.spike_clusters, spike_clusters)

    # Redo.
    up = clustering.redo()
    assert up.deleted == [2]
    assert up.added == [8, 9]
    assert set(up.descendants) == set([(2, 8), (2, 9)])
    ae(clustering.spike_clusters, [8, 5, 3, 9, 7, 5, 9])

    # Second split: just replace cluster 8 by 10 (1 spike in it).
    up = clustering.split([0])
    assert up.deleted == [8]
    assert up.added == [10]
    assert set(up.descendants) == set([(8, 10)])
    ae(clustering.spike_clusters, [10, 5, 3, 9, 7, 5, 9])

    # Undo again.
    up = clustering.undo()
    assert up.deleted == [10]
    assert up.added == [8]
    assert set(up.descendants) == set([(10, 8)])
    ae(clustering.spike_clusters, [8, 5, 3, 9, 7, 5, 9])


def test_clustering_merge():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    clustering = Clustering(spike_clusters)
    spk0 = clustering.spikes_per_cluster[0]
    spk1 = clustering.spikes_per_cluster[1]

    checkpoints = {}

    def _checkpoint():
        index = len(checkpoints)
        checkpoints[index] = clustering.spike_clusters.copy()

    def _assert_is_checkpoint(index):
        ae(clustering.spike_clusters, checkpoints[index])

    def _assert_spikes(clusters):
        ae(info.spike_ids, _spikes_in_clusters(spike_clusters, clusters))

    @connect(sender=clustering)
    def on_request_undo_state(sender, up):
        return 'hello'

    # Checkpoint 0.
    _checkpoint()
    _assert_is_checkpoint(0)

    # Checkpoint 1.
    info = clustering.merge([0, 1], 11)
    _checkpoint()
    _assert_spikes([11])
    ae(clustering.spikes_per_cluster[11], np.sort(np.r_[spk0, spk1]))
    assert 0 not in clustering.spikes_per_cluster
    assert info.added == [11]
    assert info.deleted == [0, 1]
    _assert_is_checkpoint(1)

    # Checkpoint 2.
    info = clustering.merge([2, 3], 12)
    _checkpoint()
    _assert_spikes([12])
    assert info.added == [12]
    assert info.deleted == [2, 3]
    assert info.history is None
    assert info.undo_state is None  # undo_state is only returned when undoing.
    _assert_is_checkpoint(2)

    # Undo once.
    info = clustering.undo()
    assert info.added == [2, 3]
    assert info.deleted == [12]
    assert info.history == 'undo'
    assert info.undo_state == ['hello']
    _assert_is_checkpoint(1)
    ae(clustering.spikes_per_cluster[11], np.sort(np.r_[spk0, spk1]))

    # Redo.
    info = clustering.redo()
    _assert_spikes([12])
    assert info.added == [12]
    assert info.deleted == [2, 3]
    assert info.history == 'redo'
    assert info.undo_state is None
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
    assert info.history is None
    _assert_is_checkpoint(3)

    # One more merge.
    info = clustering.merge([8, 7])  # merged to 14
    _checkpoint()
    _assert_spikes([14])
    assert info.added == [14]
    assert info.deleted == [7, 8]
    assert info.history is None
    _assert_is_checkpoint(4)

    # Now we undo.
    info = clustering.undo()
    assert info.added == [7, 8]
    assert info.deleted == [14]
    assert info.history == 'undo'
    _assert_is_checkpoint(3)

    # We merge again.
    # NOTE: 14 has been wasted, move to 15: necessary to avoid explicit cache
    # invalidation when caching clusterid-based functions.
    assert clustering.new_cluster_id() == 15
    assert any(clustering.spike_clusters == 13)
    assert all(clustering.spike_clusters != 14)
    info = clustering.merge([8, 7], 15)
    _assert_spikes([15])
    assert info.added == [15]
    assert info.deleted == [7, 8]
    assert info.history is None
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

    @connect(sender=clustering)
    def on_request_undo_state(sender, up):
        return 'hello'

    # Checkpoint 0.
    _checkpoint()
    _assert_is_checkpoint(0)

    my_spikes_1 = np.unique(np.random.randint(low=0, high=n_spikes, size=5))
    my_spikes_2 = np.unique(np.random.randint(low=0, high=n_spikes, size=10))
    my_spikes_3 = np.unique(np.random.randint(low=0, high=n_spikes, size=1000))
    my_spikes_4 = np.arange(n_spikes - 5)

    # Edge cases.
    clustering.assign([])
    with raises(ValueError):
        clustering.merge([], 1)

    # Checkpoint 1.
    info = clustering.split(my_spikes_1)
    _checkpoint()
    assert info.description == 'assign'
    assert 10 in info.added
    assert info.history is None
    _assert_is_checkpoint(1)

    # Checkpoint 2.
    info = clustering.split(my_spikes_2)
    assert info.description == 'assign'
    assert info.history is None
    _checkpoint()
    _assert_is_checkpoint(2)

    # Checkpoint 3.
    info = clustering.assign(my_spikes_3)
    assert info.description == 'assign'
    assert info.history is None
    assert info.undo_state is None
    _checkpoint()
    _assert_is_checkpoint(3)

    # Undo checkpoint 3.
    info = clustering.undo()
    assert info.description == 'assign'
    assert info.history == 'undo'
    assert info.undo_state == ['hello']
    _checkpoint()
    _assert_is_checkpoint(2)

    # Checkpoint 4.
    info = clustering.assign(my_spikes_4)
    assert info.description == 'assign'
    assert info.history is None
    _checkpoint(4)
    assert len(info.deleted) >= 2
    _assert_is_checkpoint(4)


def test_clustering_new_id():
    spike_clusters = 10 * np.ones(6, dtype=np.int32)
    spike_clusters[2:4] = 20
    spike_clusters[4:6] = 30
    clustering = Clustering(spike_clusters)
    clustering.split(list(range(1, 5)))
    ae(clustering.spike_clusters, [32, 31, 31, 31, 31, 33])
    assert clustering.new_cluster_id() == 34


def test_clustering_long():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_clusters_base = spike_clusters.copy()

    # Instantiate a Clustering instance.
    clustering = Clustering(spike_clusters)
    ae(clustering.spike_clusters, spike_clusters)

    # Test clustering.spikes_in_clusters() function.:
    assert np.all(spike_clusters[clustering.spikes_in_clusters([5])] == 5)

    # Test cluster ids.
    ae(clustering.cluster_ids, np.arange(n_clusters))

    assert clustering.new_cluster_id() == n_clusters
    assert clustering.n_clusters == n_clusters

    # Updating a cluster, method 1.
    spike_clusters_new = spike_clusters.copy()
    spike_clusters_new[:10] = 100
    clustering.spike_clusters[:] = spike_clusters_new[:]
    # Need to update explicitely.
    clustering._new_cluster_id = 101
    clustering._update_cluster_ids()
    ae(clustering.cluster_ids, np.r_[np.arange(n_clusters), 100])

    # Updating a cluster, method 2.
    clustering.spike_clusters[:] = spike_clusters_base[:]
    clustering.spike_clusters[:10] = 100
    # HACK: need to update manually here.
    clustering._new_cluster_id = 101
    ae(clustering.cluster_ids, np.r_[np.arange(n_clusters), 100])

    # Assign.
    new_cluster = 101
    clustering.assign(np.arange(0, 10), new_cluster)
    assert new_cluster in clustering.cluster_ids
    assert np.all(clustering.spike_clusters[:10] == new_cluster)

    # Merge.
    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [2, 3]))[0]
    info = clustering.merge([2, 3])
    my_spikes = info.spike_ids
    ae(my_spikes, my_spikes_0)
    assert (new_cluster + 1) in clustering.cluster_ids
    assert np.all(clustering.spike_clusters[my_spikes] == (new_cluster + 1))

    # Merge to a given cluster.
    clustering.spike_clusters[:] = spike_clusters_base[:]
    clustering._new_cluster_id = 11

    my_spikes_0 = np.nonzero(np.in1d(clustering.spike_clusters, [4, 6]))[0]
    info = clustering.merge([4, 6], 11)
    my_spikes = info.spike_ids
    ae(my_spikes, my_spikes_0)
    assert 11 in clustering.cluster_ids
    assert np.all(clustering.spike_clusters[my_spikes] == 11)

    # Split.
    my_spikes = [1, 3, 5]
    clustering.split(my_spikes)
    assert np.all(clustering.spike_clusters[my_spikes] == 12)

    # Assign.
    clusters = [0, 1, 2]
    clustering.assign(my_spikes, clusters)
    clu = clustering.spike_clusters[my_spikes]
    ae(clu - clu[0], clusters)
