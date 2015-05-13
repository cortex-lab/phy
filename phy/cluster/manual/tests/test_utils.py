# -*- coding: utf-8 -*-

"""Tests of manual clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.array import _unique
from .._utils import (ClusterMetadataUpdater,
                      _spikes_in_clusters,
                      _spikes_per_cluster,
                      _flatten_spikes_per_cluster,
                      _concatenate_per_cluster_arrays,
                      _subset_spikes_per_cluster,
                      )
from ....io.mock import artificial_spike_clusters
from ....io.kwik_model import ClusterMetadata


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


def test_subset_spikes_per_cluster():

    # 8, 11, 12, 13, 17, 18, 20
    spikes_per_cluster = {2: [11, 13, 17], 3: [8, 12], 5: [18, 20]}

    arrays = {2: [1, 3, 7], 3: [8, 2], 5: [8, 0]}

    spikes = [8, 11, 17, 18]

    spc, arrs = _subset_spikes_per_cluster(spikes_per_cluster, arrays, spikes)

    ae(spc[2], [11, 17])
    ae(spc[3], [8])
    ae(spc[5], [18])

    ae(arrs[2], [1, 7])
    ae(arrs[3], [8])
    ae(arrs[5], [8])


def test_metadata_history():
    """Test ClusterMetadataUpdater history."""

    data = {2: {'group': 2, 'color': 7}, 4: {'group': 5}}

    base_meta = ClusterMetadata(data=data)

    @base_meta.default
    def group(cluster):
        return 3

    @base_meta.default
    def color(cluster):
        return 0

    meta = ClusterMetadataUpdater(base_meta)

    # Values set in 'data'.
    assert meta.group(2) == 2
    assert meta.color(2) == 7

    # Default values.
    assert meta.group(3) == 3
    assert meta.color(3) != 7

    assert meta.group(4) == 5
    assert meta.color(4) != 7

    ###########

    meta.undo()
    meta.redo()

    # Action 1.
    info = meta.set_group(2, 20)
    assert meta.group(2) == 20
    assert info.description == 'metadata_group'
    assert info.metadata_changed == [2]

    # Action 2.
    info = meta.set_color(3, 30)
    assert meta.color(3) == 30
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Action 3.
    info = meta.set_color(2, 40)
    assert meta.color(2) == 40
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [2]

    ###########

    # Undo 3.
    info = meta.undo()
    assert meta.color(2) == 7
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [2]

    # Undo 2.
    info = meta.undo()
    assert meta.color(3) != 7
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Redo 2.
    info = meta.redo()
    assert meta.color(3) == 30
    assert meta.group(2) == 20
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Undo 2.
    info = meta.undo()
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Undo 1.
    info = meta.undo()
    assert meta.group(2) == 2
    assert info.description == 'metadata_group'
    assert info.metadata_changed == [2]

    info = meta.undo()
    assert info is None

    info = meta.undo()
    assert info is None
