# -*- coding: utf-8 -*-

"""Mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import numpy.random as nr

from ..ext import six
from ..io.experiment import BaseExperiment


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

    def metadata(self):
        return {'description': 'A mock experiment.'}

    def traces(self, recording=0):
        return artificial_traces(self.n_samples_traces, self.n_channels)

    def spike_times(self, channel_group=0):
        return artificial_spike_times(self.n_spikes)

    def spike_clusters(self, channel_group=0):
        return artificial_spike_clusters(self.n_spikes, self.n_clusters)

    def cluster_metadata(self, channel_group=0):
        """ClusterMetadata instance holding information about the clusters.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    def features(self, channel_group=0):
        """Features from a given channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def masks(self, channel_group=0):
        """Masks from a given channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def waveforms(self, channel_group=0):
        """Waveforms from a given channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def probe(self):
        """A Probe instance.

        May be implemented by child classes.

        """
        raise NotImplementedError()
