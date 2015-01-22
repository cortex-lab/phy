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
from ..cluster_metadata import ClusterMetadata


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_cluster_metadata():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters, low=2)

    meta = ClusterMetadata()
    assert meta.data is not None

    with raises(ValueError):
        assert meta[0]['group']

    # Specify spike_clusters.
    meta.spike_clusters = spike_clusters
    assert meta.spike_clusters is not None

    assert_array_equal(meta.cluster_labels, np.arange(2, 10))

    with raises(ValueError):
        assert meta[0]['group']

    assert meta[2]['color'] == 1
    assert meta[2]['group'] == 3

    with raises(ValueError):
        assert meta[10]

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
        assert meta[2]

    assert meta[10]['color'] == 1
    assert meta[10]['group'] == 3

    meta.set([10], 'color', 5)
    assert meta[10]['color'] == 5

    # WARNING: __getitem__ returns a copy so changing this has no effect.
    meta[10]['color'] == 7
    assert meta[10]['color'] == 5


def test_default_function():
    meta = ClusterMetadata([('field', lambda x: x * x)])
    meta._add_clusters([3])

    assert meta[3]['field'] == 9
