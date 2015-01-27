# -*- coding: utf-8 -*-

"""Test waveform plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy import app

from ...utils.logging import set_level
from ..waveforms import Waveforms, WaveformView
from ...electrode.mea import staggered_positions
from ...utils.array import _normalize
from ...utils.testing import show_test


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    pass


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_waveforms():
    n_clusters = 3
    n_channels = 32
    n_samples = 40
    n_spikes = 100

    channel_positions = staggered_positions(n_channels)
    channel_positions = _normalize(channel_positions)

    waveforms = .25 * np.random.randn(n_spikes, n_channels,
                                      n_samples).astype(np.float32)
    masks = np.random.rand(n_spikes, n_channels).astype(np.float32)
    cluster_colors = np.random.uniform(size=(n_clusters, 3),
                                       low=.5, high=.9).astype(np.float32)
    cluster_metadata = {cluster: {'color': color}
                        for cluster, color in enumerate(cluster_colors)}

    spike_clusters = np.random.randint(size=n_spikes,
                                       low=0,
                                       high=n_clusters).astype(np.int32)

    c = WaveformView()
    c.visual.waveforms = waveforms
    c.visual.masks = masks
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_metadata = cluster_metadata
    c.visual.channel_positions = channel_positions

    show_test(c)
