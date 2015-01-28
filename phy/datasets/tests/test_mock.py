# -*- coding: utf-8 -*-

"""Tests of mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal

from ..mock import (artificial_waveforms,
                    artificial_traces,
                    artificial_spike_clusters,
                    artificial_features,
                    artificial_masks)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_artificial():
    n_spikes = 100
    n_samples_waveforms = 32
    n_samples_traces = 50
    n_channels = 64
    n_clusters = 10
    n_features = n_channels * 3

    # Waveforms.
    waveforms = artificial_waveforms(n_spikes=n_spikes,
                                     n_samples=n_samples_waveforms,
                                     n_channels=n_channels)
    assert waveforms.shape == (n_spikes, n_samples_waveforms, n_channels)

    # Traces.
    traces = artificial_traces(n_samples=n_samples_traces,
                               n_channels=n_channels)
    assert traces.shape == (n_samples_traces, n_channels)

    # Spike clusters.
    spike_clusters = artificial_spike_clusters(n_spikes=n_spikes,
                                               n_clusters=n_clusters)
    assert spike_clusters.shape == (n_spikes,)
    assert (spike_clusters.min(), spike_clusters.max()) == (0, n_clusters - 1)
    assert_array_equal(np.unique(spike_clusters), np.arange(n_clusters))

    # Features.
    features = artificial_features(n_spikes, n_features)
    assert features.shape == (n_spikes, n_features)

    # Masks.
    masks = artificial_masks(n_spikes, n_channels)
    assert masks.shape == (n_spikes, n_channels)
