# -*- coding: utf-8 -*-

"""Mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import numpy.random as nr


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
