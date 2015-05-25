# -*- coding: utf-8 -*-

"""Test selector."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ...io.mock import artificial_spike_clusters
from ..array import _spikes_in_clusters
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
    selector.on_cluster()
    assert selector.n_spikes_max is None
    selector.n_spikes_max = None
    ae(selector.selected_spikes, [])

    # Select a few spikes.
    my_spikes = [10, 20, 30]
    selector.selected_spikes = my_spikes
    ae(selector.selected_spikes, my_spikes)

    # Check selected clusters.
    ae(selector.selected_clusters, np.unique(spike_clusters[my_spikes]))

    # Specify a maximum number of spikes.
    selector.n_spikes_max = 3
    assert selector.n_spikes_max is 3
    my_spikes = [10, 20, 30, 40]
    selector.selected_spikes = my_spikes[:3]
    ae(selector.selected_spikes, my_spikes[:3])
    selector.selected_spikes = my_spikes
    assert len(selector.selected_spikes) <= 3
    assert selector.n_spikes == len(selector.selected_spikes)
    assert np.all(np.in1d(selector.selected_spikes, my_spikes))

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
    ae(selector.selected_spikes, [])

    # Select 1 cluster.
    selector.selected_clusters = [0]
    ae(selector.selected_spikes, _spikes_in_clusters(spike_clusters, [0]))
    assert np.all(spike_clusters[selector.selected_spikes] == 0)

    # Select 2 clusters.
    selector.selected_clusters = [1, 3]
    ae(selector.selected_spikes, _spikes_in_clusters(spike_clusters, [1, 3]))
    assert np.all(np.in1d(spike_clusters[selector.selected_spikes], (1, 3)))
    assert selector.n_clusters == 2

    # Specify a maximum number of spikes.
    selector.n_spikes_max = 10
    selector.selected_clusters = [4, 2]
    assert len(selector.selected_spikes) <= (10 * 2)
    assert selector.selected_clusters == [4, 2]
    assert np.all(np.in1d(spike_clusters[selector.selected_spikes], (2, 4)))

    # Reduce the number of maximum spikes: the selection should update
    # accordingly.
    selector.n_spikes_max = 5
    assert len(selector.selected_spikes) <= 5
    assert np.all(np.in1d(spike_clusters[selector.selected_spikes], (2, 4)))


def test_selector_subset():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    selector = Selector(spike_clusters)
    selector.subset_spikes(excerpt_size=10)
    selector.subset_spikes(np.arange(n_spikes), excerpt_size=10)


def test_selector_subset_clusters():
    n_spikes = 100
    spike_clusters = np.zeros(n_spikes, dtype=np.int32)
    spike_clusters[10:15] = 1
    spike_clusters[85:90] = 1

    selector = Selector(spike_clusters)
    spc = selector.subset_spikes_clusters([0, 1], excerpt_size=10)
    counts = {_: len(spc[_]) for _ in sorted(spc.keys())}
    # TODO
    assert counts
