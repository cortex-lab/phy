# -*- coding: utf-8 -*-

"""Tests of clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.io.mock import artificial_features, artificial_masks
from ..klustakwik import cluster


#------------------------------------------------------------------------------
# Tests clustering
#------------------------------------------------------------------------------

def test_cluster(tempdir):
    n_channels = 4
    n_spikes = 100
    features = artificial_features(n_spikes, n_channels * 3)
    masks = artificial_masks(n_spikes, n_channels * 3)

    spike_clusters = cluster(features, masks, num_starting_clusters=10)
    assert len(spike_clusters) == n_spikes

    spike_clusters = cluster(features, masks, num_starting_clusters=10,
                             spike_ids=range(100))
    assert len(spike_clusters) == 100
