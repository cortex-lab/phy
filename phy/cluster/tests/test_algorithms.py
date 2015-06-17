# -*- coding: utf-8 -*-

"""Tests of clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ...utils.tempdir import TemporaryDirectory
from ..algorithms import cluster
from ...io.kwik import KwikModel
from ...io.kwik.mock import create_mock_kwik


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_cluster():
    n_spikes = 100
    with TemporaryDirectory() as tempdir:
        filename = create_mock_kwik(tempdir,
                                    n_clusters=1,
                                    n_spikes=n_spikes,
                                    n_channels=8,
                                    n_features_per_channel=3,
                                    n_samples_traces=5000)
        model = KwikModel(filename)

        spike_clusters = cluster(model, num_starting_clusters=10)
        assert len(spike_clusters) == n_spikes

        spike_clusters = cluster(model, num_starting_clusters=10,
                                 spike_ids=range(100))
        assert len(spike_clusters) == 100
