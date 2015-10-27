# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.io.mock import (artificial_waveforms,
                         artificial_spike_clusters,
                         artificial_masks,
                         )
from phy.electrode.mea import staggered_positions
from ..views import WaveformView


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _show(qtbot, view, stop=False):
    view.show()
    qtbot.waitForWindowShown(view.native)
    if stop:  # pragma: no cover
        qtbot.stop()
    view.close()


#------------------------------------------------------------------------------
# Test views
#------------------------------------------------------------------------------

def test_waveform_view(qtbot):
    n_spikes = 20
    n_samples = 30
    n_channels = 40
    n_clusters = 3

    waveforms = artificial_waveforms(n_spikes, n_samples, n_channels)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    channel_positions = staggered_positions(n_channels)

    # Create the view.
    v = WaveformView(waveforms=waveforms,
                     masks=masks,
                     spike_clusters=spike_clusters,
                     channel_positions=channel_positions,
                     )

    # Select some spikes.
    spike_ids = np.arange(5)
    cluster_ids = np.unique(spike_clusters[spike_ids])
    v.on_select(cluster_ids, spike_ids)

    # Show the view.
    v.show()
    qtbot.waitForWindowShown(v.native)

    # Select other spikes.
    spike_ids = np.arange(2, 10)
    cluster_ids = np.unique(spike_clusters[spike_ids])
    v.on_select(cluster_ids, spike_ids)

    # qtbot.stop()
    v.close()
