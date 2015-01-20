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
from ..selector import Selector


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_selector():
    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    selector = Selector(spike_clusters)
    assert selector.n_spikes_max is None
    selector.n_spikes_max = None
    assert_array_equal(selector.selected_spikes, [])

    # Select a few spikes.
    myspikes = [10, 20, 30]
    selector.selected_spikes = myspikes
    assert_array_equal(selector.selected_spikes, myspikes)

    # Check selected clusters.
    assert_array_equal(selector.selected_clusters,
                       np.unique(spike_clusters[myspikes]))

    # Specify a maximum number of spikes.
    selector.n_spikes_max = 3
    assert selector.n_spikes_max is 3
    myspikes = [10, 20, 30, 40]
    selector.selected_spikes = myspikes[:3]
    assert_array_equal(selector.selected_spikes, myspikes[:3])
    selector.selected_spikes = myspikes
    assert_array_equal(selector.selected_spikes, myspikes[:3])
