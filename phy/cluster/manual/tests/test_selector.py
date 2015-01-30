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
from .._utils import _spikes_in_clusters
from ..selector import Selector


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_selector_spikes():
    """Test selecting spikes."""
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    selector = Selector(spike_clusters)
    selector.update()
    assert selector.n_spikes_max is None
    selector.n_spikes_max = None
    assert_array_equal(selector.selected_spikes, [])

    # Select a few spikes.
    my_spikes = [10, 20, 30]
    selector.selected_spikes = my_spikes
    assert_array_equal(selector.selected_spikes, my_spikes)

    # Check selected clusters.
    assert_array_equal(selector.selected_clusters,
                       np.unique(spike_clusters[my_spikes]))

    # Specify a maximum number of spikes.
    selector.n_spikes_max = 3
    assert selector.n_spikes_max is 3
    my_spikes = [10, 20, 30, 40]
    selector.selected_spikes = my_spikes[:3]
    assert_array_equal(selector.selected_spikes, my_spikes[:3])
    selector.selected_spikes = my_spikes
    assert_array_equal(selector.selected_spikes, [10, 30, 40])

    # Check that this doesn't raise any error.
    selector.selected_clusters = [100]
    selector.selected_spikes = []


def test_selector_clusters():
    """Test selecting clusters."""
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    selector = Selector(spike_clusters)
    selector.selected_clusters = []
    assert_array_equal(selector.selected_spikes, [])

    # Select 1 cluster.
    selector.selected_clusters = [0]
    assert_array_equal(selector.selected_spikes,
                       _spikes_in_clusters(spike_clusters, [0]))
    assert np.all(spike_clusters[selector.selected_spikes] == 0)

    # Select 2 clusters.
    selector.selected_clusters = [1, 3]
    assert_array_equal(selector.selected_spikes,
                       _spikes_in_clusters(spike_clusters, [1, 3]))
    assert np.all(np.in1d(spike_clusters[selector.selected_spikes], (1, 3)))

    # Specify a maximum number of spikes.
    selector.n_spikes_max = 10
    selector.selected_clusters = [2, 4]
    assert len(selector.selected_spikes) <= 10
    assert np.all(np.in1d(spike_clusters[selector.selected_spikes], (2, 4)))

    # Reduce the number of maximum spikes: the selection should update
    # accordingly.
    selector.n_spikes_max = 5
    assert len(selector.selected_spikes) <= 5
    assert np.all(np.in1d(spike_clusters[selector.selected_spikes], (2, 4)))
