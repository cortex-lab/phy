# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.io.array import _index_of
from phy.electrode.mea import linear_positions
from phy.plot import BoxedView, _get_linear_x
from phy.plot.visuals import _get_data_bounds
from phy.plot.transform import Range

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


# Default color map for the selected clusters.
_COLORMAP = np.array([[8, 146, 252],
                      [255, 2, 2],
                      [240, 253, 2],
                      [228, 31, 228],
                      [2, 217, 2],
                      [255, 147, 2],
                      [212, 150, 70],
                      [205, 131, 201],
                      [201, 172, 36],
                      [150, 179, 62],
                      [95, 188, 122],
                      [129, 173, 190],
                      [231, 107, 119],
                      ])


def _selected_clusters_colors(n_clusters=None):
    if n_clusters is None:
        n_clusters = _COLORMAP.shape[0]
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

class WaveformView(BoxedView):
    def __init__(self,
                 waveforms=None,
                 masks=None,
                 spike_clusters=None,
                 channel_positions=None,
                 ):
        """

        The channel order in waveforms needs to correspond to the one
        in channel_positions.

        """

        # Initialize the view.
        if channel_positions is None:
            channel_positions = linear_positions(self.n_channels)
        bounds = _get_data_bounds(None, channel_positions)
        channel_positions = Range(from_bounds=bounds).apply(channel_positions)
        channel_positions *= .75

        self.box_size = (.1, .1)
        bs = np.array([self.box_size])
        box_bounds = np.c_[channel_positions - bs / 2.,
                           channel_positions + bs / 2.,
                           ]
        super(WaveformView, self).__init__(box_bounds)

        # Waveforms.
        assert waveforms.ndim == 3
        self.n_spikes, self.n_samples, self.n_channels = waveforms.shape
        self.waveforms = waveforms

        # Masks.
        if masks is None:
            masks = np.ones((self.n_spikes, self.n_channels), dtype=np.float32)
        assert masks.ndim == 2
        assert masks.shape == (self.n_spikes, self.n_channels)
        self.masks = masks

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters

        # Channel positions.
        assert channel_positions.shape == (self.n_channels, 2)
        self.channel_positions = channel_positions

    def on_select(self, cluster_ids, spike_ids):
        n_clusters = len(cluster_ids)
        n_spikes = len(spike_ids)
        if n_spikes == 0:
            return

        # Relative spike clusters.
        # NOTE: the order of the clusters in cluster_ids matters.
        # It will influence the relative index of the clusters, which
        # in return influence the depth.
        spike_clusters = self.spike_clusters[spike_ids]
        assert np.all(np.in1d(spike_clusters, cluster_ids))
        spike_clusters_rel = _index_of(spike_clusters, cluster_ids)

        # Fetch the waveforms.
        w = self.waveforms[spike_ids]
        colors = _selected_clusters_colors(n_clusters)
        t = _get_linear_x(n_spikes, self.n_samples)

        # Get the colors.
        color = colors[spike_clusters_rel[spike_ids]]
        # Alpha channel.
        color = np.c_[color, np.ones((n_spikes, 1))]
        # TODO: depth

        # Plot all waveforms.
        for ch in range(self.n_channels):
            self[ch].plot(x=t, y=w[:, :, ch], color=color)

        self.build()
        self.update()

    def on_cluster(self, up):
        pass

    def on_mouse_move(self, e):
        pass

    def on_key_press(self, e):
        pass

    def attach_to_gui(self, gui):
        gui.add_view(self)

        # TODO: make sure the GUI emits these events
        gui.connect(self.on_select)
        gui.connect(self.on_cluster)
