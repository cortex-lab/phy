# -*- coding: utf-8 -*-

"""Mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import numpy.random as nr

from ..utils._color import _random_color
from ..utils.array import _unique, _spikes_per_cluster
from .base import BaseModel, ClusterMetadata
from ..electrode.mea import MEA, staggered_positions


#------------------------------------------------------------------------------
# Artificial data
#------------------------------------------------------------------------------

def artificial_waveforms(n_spikes=None, n_samples=None, n_channels=None):
    # TODO: more realistic waveforms.
    return .25 * nr.normal(size=(n_spikes, n_samples, n_channels))


def artificial_features(*args):
    return .25 * nr.normal(size=args)


def artificial_masks(n_spikes=None, n_channels=None):
    masks = nr.uniform(size=(n_spikes, n_channels))
    masks[masks < .25] = 0
    return masks


def artificial_traces(n_samples, n_channels):
    # TODO: more realistic traces.
    return .25 * nr.normal(size=(n_samples, n_channels))


def artificial_spike_clusters(n_spikes, n_clusters, low=0):
    return nr.randint(size=n_spikes, low=low, high=max(1, n_clusters))


def artificial_spike_samples(n_spikes, max_isi=50):
    return np.cumsum(nr.randint(low=0, high=max_isi, size=n_spikes))


def artificial_correlograms(n_clusters, n_samples):
    return nr.uniform(size=(n_clusters, n_clusters, n_samples))


#------------------------------------------------------------------------------
# Artificial Model
#------------------------------------------------------------------------------

class MockModel(BaseModel):
    n_channels = 28
    n_features_per_channel = 2
    n_features = 28 * n_features_per_channel
    n_spikes = 1000
    n_samples_traces = 20000
    n_samples_waveforms = 40
    n_clusters = 10
    sample_rate = 20000.

    def __init__(self, n_spikes=None, n_clusters=None):
        super(MockModel, self).__init__()
        if n_spikes is not None:
            self.n_spikes = n_spikes
        if n_clusters is not None:
            self.n_clusters = n_clusters
        self.name = 'mock'
        self._clustering = 'main'
        nfpc = self.n_features_per_channel
        self._metadata = {'description': 'A mock model.',
                          'n_features_per_channel': nfpc}
        self._cluster_metadata = ClusterMetadata()

        @self._cluster_metadata.default
        def group(cluster):
            if cluster <= 2:
                return cluster
            # Default group is unsorted.
            return 3

        @self._cluster_metadata.default
        def color(cluster):
            return _random_color()

        positions = staggered_positions(self.n_channels)
        self._probe = MEA(channels=self.channels, positions=positions)
        self._traces = artificial_traces(self.n_samples_traces,
                                         self.n_channels)
        self._spike_clusters = artificial_spike_clusters(self.n_spikes,
                                                         self.n_clusters)
        self._spike_ids = np.arange(self.n_spikes).astype(np.int64)
        self._spikes_per_cluster = _spikes_per_cluster(self._spike_ids,
                                                       self._spike_clusters)
        self._spike_samples = artificial_spike_samples(self.n_spikes, 30)
        assert self._spike_samples[-1] < self.n_samples_traces
        self._features = artificial_features(self.n_spikes, self.n_features)
        self._masks = artificial_masks(self.n_spikes, self.n_channels)
        self._features_masks = np.dstack((self._features,
                                          np.repeat(self._masks,
                                                    nfpc,
                                                    axis=1)))
        self._waveforms = artificial_waveforms(self.n_spikes,
                                               self.n_samples_waveforms,
                                               self.n_channels)

    @property
    def channels(self):
        return np.arange(self.n_channels)

    @property
    def channel_order(self):
        return self.channels

    @property
    def metadata(self):
        return self._metadata

    @property
    def traces(self):
        return self._traces

    @property
    def spike_samples(self):
        return self._spike_samples

    @property
    def spike_clusters(self):
        return self._spike_clusters

    @property
    def spikes_per_cluster(self):
        return self._spikes_per_cluster

    def update_spikes_per_cluster(self, spc):
        self._spikes_per_cluster = spc

    @property
    def spike_ids(self):
        return self._spike_ids

    @property
    def cluster_ids(self):
        return _unique(self._spike_clusters)

    @property
    def cluster_metadata(self):
        return self._cluster_metadata

    @property
    def features(self):
        return self._features

    @property
    def masks(self):
        return self._masks

    @property
    def features_masks(self):
        return self._features_masks

    @property
    def waveforms(self):
        return self._waveforms

    @property
    def probe(self):
        return self._probe
