# -*- coding: utf-8 -*-

"""Test waveform plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

import numpy as np

from ..waveforms import WaveformView
from ...utils._color import _random_color
from ...io.mock import (artificial_waveforms, artificial_masks,
                        artificial_spike_clusters)
from ...electrode.mea import staggered_positions
from ...utils.testing import show_test


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------


def _test_waveforms(n_spikes=None, n_clusters=None):
    n_channels = 32
    n_samples = 40

    channel_positions = staggered_positions(n_channels)

    waveforms = artificial_waveforms(n_spikes, n_samples, n_channels)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    c = WaveformView()
    c.visual.waveforms = waveforms
    # Test depth.
    # masks[n_spikes//2:, ...] = 0
    # Test position of masks.
    # masks[:, :n_channels // 2] = 0
    # masks[:, n_channels // 2:] = 1
    c.visual.masks = masks
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_colors = np.array([_random_color()
                                        for _ in range(n_clusters)])
    c.visual.channel_positions = channel_positions
    c.visual.channel_order = np.arange(1, n_channels + 1)

    @c.connect
    def on_channel_click(e):
        print(e.channel_id, e.key)

    show_test(c)


def test_waveforms_empty():
    _test_waveforms(n_spikes=0, n_clusters=0)


def test_waveforms_full():
    _test_waveforms(n_spikes=100, n_clusters=3)
