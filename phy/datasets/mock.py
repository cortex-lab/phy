# -*- coding: utf-8 -*-

"""Mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import numpy.random as nr

from ..ext import six


#------------------------------------------------------------------------------
# Artificial data
#------------------------------------------------------------------------------

def artificial_waveforms(n_spikes=None, n_samples=None, n_channels=None):
    # TODO: more realistic waveforms.
    return .25 * nr.normal(size=(n_spikes, n_samples, n_channels))


def artificial_traces(n_samples, n_channels):
    # TODO: more realistic traces.
    return .25 * nr.normal(size=(n_samples, n_channels))


def artificial_spike_clusters(n_spikes, n_clusters):
    return nr.randint(size=n_spikes, low=0, high=n_clusters)
