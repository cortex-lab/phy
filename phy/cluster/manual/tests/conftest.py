# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

import numpy as np

from phy.cluster.manual.controller import Controller
from phy.electrode.mea import staggered_positions
from phy.io.array import (get_closest_clusters,
                          _spikes_in_clusters,
                          )
from phy.io.mock import (artificial_waveforms,
                         artificial_features,
                         artificial_masks,
                         artificial_traces,
                         )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def cluster_ids():
    return [0, 1, 2, 10, 11, 20, 30]
    #       i, g, N,  i,  g,  N, N


@fixture
def cluster_groups():
    return {0: 'noise', 1: 'good', 10: 'mua', 11: 'good'}


@fixture
def quality():
    def quality(c):
        return c
    return quality


@fixture
def similarity(cluster_ids):
    sim = lambda c, d: (c * 1.01 + d)

    def similarity(c):
        return get_closest_clusters(c, cluster_ids, sim)
    return similarity


class MockController(Controller):
    def _init_data(self):
        self.cache_dir = self.config_dir
        self.n_samples_waveforms = 31
        self.n_samples_t = 20000
        self.n_channels = 11
        self.n_clusters = 4
        self.n_spikes_per_cluster = 200
        n_spikes_total = self.n_clusters * self.n_spikes_per_cluster
        n_features_per_channel = 4

        self.n_channels = self.n_channels
        self.n_spikes = n_spikes_total
        self.sample_rate = 20000.
        self.duration = self.n_samples_t / float(self.sample_rate)
        self.spike_times = np.arange(0, self.duration,
                                     5000. / (self.sample_rate *
                                              self.n_spikes_per_cluster))
        self.spike_clusters = np.repeat(np.arange(self.n_clusters),
                                        self.n_spikes_per_cluster)
        assert len(self.spike_times) == len(self.spike_clusters)
        self.cluster_ids = np.unique(self.spike_clusters)
        self.channel_positions = staggered_positions(self.n_channels)
        self.channel_order = np.arange(self.n_channels)

        sc = self.spike_clusters
        self.spikes_per_cluster = lambda c: _spikes_in_clusters(sc, [c])
        self.spike_count = lambda c: len(self.spikes_per_cluster(c))
        self.n_features_per_channel = n_features_per_channel
        self.cluster_groups = {c: None for c in range(self.n_clusters)}

        self.all_traces = artificial_traces(self.n_samples_t, self.n_channels)
        self.all_masks = artificial_masks(n_spikes_total, self.n_channels)
        self.all_waveforms = artificial_waveforms(n_spikes_total,
                                                  self.n_samples_waveforms,
                                                  self.n_channels)
        self.all_features = artificial_features(n_spikes_total,
                                                self.n_channels,
                                                self.n_features_per_channel)
