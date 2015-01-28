# -*- coding: utf-8 -*-

"""Mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import numpy.random as nr

from ..ext import six
from ..io.experiment import BaseExperiment
from ..cluster.manual.cluster_metadata import ClusterMetadata
from ..electrode.mea import MEA, staggered_positions


#------------------------------------------------------------------------------
# Artificial data
#------------------------------------------------------------------------------

def artificial_waveforms(n_spikes=None, n_samples=None, n_channels=None):
    # TODO: more realistic waveforms.
    return .25 * nr.normal(size=(n_spikes, n_samples, n_channels))


def artificial_features(n_spikes=None, n_features=None):
    return .25 * nr.normal(size=(n_spikes, n_features))


def artificial_masks(n_spikes=None, n_channels=None):
    return nr.uniform(size=(n_spikes, n_channels))


def artificial_traces(n_samples, n_channels):
    # TODO: more realistic traces.
    return .25 * nr.normal(size=(n_samples, n_channels))


def artificial_spike_clusters(n_spikes, n_clusters, low=0):
    return nr.randint(size=n_spikes, low=low, high=n_clusters)


def artificial_spike_times(n_spikes):
    # TODO: switch from sample to seconds in the way spike times are
    # represented throughout the package.
    return np.cumsum(nr.randint(low=0, high=100, size=n_spikes))


#------------------------------------------------------------------------------
# Artificial Experiment
#------------------------------------------------------------------------------

class MockExperiment(BaseExperiment):
    n_channels = 32
    n_spikes = 1000
    n_samples_traces = 20000
    n_samples_waveforms = 40

    def __init__(self):
        self._metadata = {'description': 'A mock experiment.'}
        self._cluster_metadata = ClusterMetadata()
        self._probe = MEA(positions=staggered_positions(self.n_channels))
        self._traces = artificial_traces(self.n_samples_traces,
                                         self.n_channels)
        self._spike_clusters = artificial_spike_clusters(self.n_spikes,
                                                         self.n_clusters)
        self._spike_times = artificial_spike_times(self.n_spikes)
        self._features = artificial_features(self.n_spikes, self.n_features)
        self._masks = artificial_masks(self.n_spikes, self.n_channels)
        self._waveforms = artificial_waveforms(self.n_spikes, self.n_samples,
                                               self.n_channels)

    def metadata(self):
        return self._metadata

    def traces(self, recording=0):
        return self._traces

    def spike_times(self, channel_group=0):
        return self._spike_times

    def spike_clusters(self, channel_group=0):
        return self._spike_clusters

    def cluster_metadata(self, channel_group=0):
        return self._cluster_metadata

    def features(self, channel_group=0):
        return self._features

    def masks(self, channel_group=0):
        return self._masks

    def waveforms(self, channel_group=0):
        return self._waveforms

    @property
    def probe(self):
        return self._probe
