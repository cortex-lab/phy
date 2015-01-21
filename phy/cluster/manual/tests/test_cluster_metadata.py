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
from ..cluster_metadata import ClusterMetadata


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_cluster_metadata():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters, low=2)

    meta = ClusterMetadata()

    with raises(ValueError):
        assert meta[0]['group']

    meta.spike_clusters = spike_clusters

    with raises(ValueError):
        assert meta[0]['group']

    assert meta[2]['color'] == 1
    assert meta[2]['group'] == 3

    spike_clusters[spike_clusters == 2] = 10
    meta.update()

    with raises(ValueError):
        assert meta[2]
