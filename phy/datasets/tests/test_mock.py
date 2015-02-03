# -*- coding: utf-8 -*-

"""Tests of mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ...electrode.mea import MEA
from ..mock import (artificial_waveforms,
                    artificial_traces,
                    artificial_spike_clusters,
                    artificial_features,
                    artificial_masks,
                    MockModel)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _test_artificial(n_spikes=None, n_clusters=None):
    n_samples_waveforms = 32
    n_samples_traces = 50
    n_channels = 35
    n_features = n_channels * 2

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
    if n_clusters >= 1:
        assert (spike_clusters.min(), spike_clusters.max()) == \
               (0, n_clusters - 1)
    assert_array_equal(np.unique(spike_clusters), np.arange(n_clusters))

    # Features.
    features = artificial_features(n_spikes, n_features)
    assert features.shape == (n_spikes, n_features)

    # Masks.
    masks = artificial_masks(n_spikes, n_channels)
    assert masks.shape == (n_spikes, n_channels)


def test_artificial():
    _test_artificial(n_spikes=100, n_clusters=10)
    _test_artificial(n_spikes=0, n_clusters=0)


def test_mock_model():
    exp = MockModel()

    assert exp.metadata['description'] == 'A mock model.'
    assert exp.traces.ndim == 2
    assert exp.spike_times.ndim == 1
    assert exp.spike_clusters.ndim == 1
    assert len(exp.cluster_metadata[3]['color']) == 3
    assert exp.features.ndim == 2
    assert exp.masks.ndim == 2
    assert exp.waveforms.ndim == 3

    assert isinstance(exp.probe, MEA)
    with raises(NotImplementedError):
        exp.save()
