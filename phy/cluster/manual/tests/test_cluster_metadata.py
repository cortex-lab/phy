# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ....datasets.mock import artificial_spike_clusters
from ....ext.six import itervalues, iterkeys
from ..cluster_metadata import _cluster_info_structure, ClusterMetadata


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_structure():
    """Test the structure holding all cluster metadata."""
    data = _cluster_info_structure([('a', 1), ('b', 2)])

    assert isinstance(data[3], dict)
    assert data[3]['a'] == 1
    assert data[3]['b'] == 2

    data[3]['b'] = 10
    assert data[3]['b'] == 10

    with raises(KeyError):
        data[3]['c']


def tAest_default_function():
    meta = ClusterMetadata([('field', lambda x: x * x)])
    meta._add_clusters([3])

    assert meta[3]['field'] == 9


def tAest_cluster_metadata():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters, low=2)

    meta = ClusterMetadata()
    assert meta.data is not None

    with raises(ValueError):
        meta[0]['group']

    # Specify spike_clusters.
    meta.spike_clusters = spike_clusters
    assert meta.spike_clusters is not None

    assert_array_equal(meta.cluster_labels, np.arange(2, 10))

    with raises(ValueError):
        meta[0]['group']

    assert meta[2]['color'] == 1
    assert meta[2]['group'] == 3

    with raises(ValueError):
        meta[10]

    # Change a cluster.
    spike_clusters[spike_clusters == 2] = 10

    assert_array_equal(meta.cluster_labels, np.arange(2, 10))
    assert_array_equal(list(itervalues(meta['color'])), np.ones(8))
    assert_array_equal(list(iterkeys(meta['group'])), np.arange(2, 10))

    meta.update()

    assert_array_equal(meta.cluster_labels, np.arange(3, 11))
    assert_array_equal(list(itervalues(meta['color'])), np.ones(8))
    assert_array_equal(list(iterkeys(meta['group'])), np.arange(3, 11))
    assert_array_equal(list(itervalues(meta['group'])), 3 * np.ones(8))

    with raises(ValueError):
        meta[2]

    assert meta[10]['color'] == 1
    assert meta[10]['group'] == 3

    meta.set([10], 'color', 5)
    assert meta[10]['color'] == 5

    # Alternative __setitem__ syntax.
    meta[[10, 11], 'color'] = 5
    assert meta[10]['color'] == 5
    assert meta[11]['color'] == 5

    meta.set([10, 11], 'color', [6, 7])
    assert meta[10]['color'] == 6
    # Alternative syntax with __getitem__.
    assert meta[11, 'color'] == 7
    assert meta[[10, 11], 'color'][10] == 6

    # WARNING: __getitem__ returns a copy so changing this has no effect.
    meta[10]['color'] == 10
    assert meta[10]['color'] == 6


def tAest_metadata_history():
    """Test ClusterMetadata history."""

    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters, low=2)

    data = {2: {'group': 2, 'color': 7}, 4: {'group': 5}}

    meta = ClusterMetadata(data=data)
    meta.spike_clusters = spike_clusters
    assert_array_equal(meta.cluster_labels, np.arange(2, 10))

    # Values set in 'data'.
    assert meta.get(2, 'group') == 2
    assert meta.get(2, 'color') == 7

    # Default values.
    assert meta.get(3, 'group') == 3
    assert meta.get(3, 'color') == 1

    assert meta.get(4, 'group') == 5
    assert meta.get(4, 'color') == 1

    ###########

    # Action 1.
    meta.set(2, 'group', 20)
    assert meta.get(2, 'group') == 20

    # Action 2.
    meta.set(3, 'color', 30)
    assert meta.get(3, 'color') == 30

    # Action 3.
    meta.set(2, 'color', 40)
    assert meta.get(2, 'color') == 40

    ###########

    # Undo 3.
    meta.undo()
    assert meta.get(2, 'color') == 7

    # Undo 2.
    meta.undo()
    assert meta.get(3, 'color') == 1

    # Redo 2.
    meta.redo()
    assert meta.get(3, 'color') == 30
    assert meta.get(2, 'group') == 20

    # Undo 2.
    meta.undo()
    # Undo 1.
    meta.undo()
    assert meta.get(2, 'group') == 2
