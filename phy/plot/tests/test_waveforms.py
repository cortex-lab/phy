# -*- coding: utf-8 -*-

"""Test waveform plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy import app

from ...utils.logging import set_level
from ..waveforms import Waveforms, WaveformView
from .._color import _random_colors
from ...datasets.mock import (artificial_waveforms, artificial_masks,
                              artificial_spike_clusters)
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


def _test_waveforms(n_spikes=None, n_clusters=None):
    n_channels = 32
    n_samples = 40

    channel_positions = staggered_positions(n_channels)
    channel_positions = _normalize(channel_positions)

    waveforms = artificial_waveforms(n_spikes, n_samples,
                                     n_channels).astype(np.float32)
    masks = artificial_masks(n_spikes, n_channels).astype(np.float32)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    cluster_colors = _random_colors(n_clusters).astype(np.float32)
    cluster_metadata = {cluster: {'color': color}
                        for cluster, color in enumerate(cluster_colors)}

    c = WaveformView()
    c.visual.waveforms = waveforms
    c.visual.masks = masks
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_metadata = cluster_metadata
    c.visual.channel_positions = channel_positions

    show_test(c)


def test_waveforms_empty():
    _test_waveforms(n_spikes=0, n_clusters=0)


def test_waveforms():
    _test_waveforms(n_spikes=100, n_clusters=3)
