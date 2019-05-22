# -*- coding: utf-8 -*-

"""Template view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot import _get_linear_x
from phylib.io.array import _index_of
from phylib.utils import emit
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

class TemplateView(ManualClusteringView):
    _default_position = 'right'
    cluster_ids = ()

    default_shortcuts = {
    }

    def __init__(self, templates=None, channel_ids=None, cluster_ids=None,
                 cluster_color_selector=None):
        super(TemplateView, self).__init__()
        self.cluster_color_selector = cluster_color_selector
        # Full list of channels.
        self.channel_ids = channel_ids
        # Full list of clusters.
        self.cluster_ids = cluster_ids
        # Total number of channels.
        self.n_channels = len(channel_ids)
        self.canvas.set_layout('grid', box_bounds=[[-1, -1, +1, +1]], has_clip=False)
        self.templates = templates

    def _get_data_bounds(self, bunchs):
        m = np.median([b.template.min() for b in bunchs.values()])
        M = np.median([b.template.max() for b in bunchs.values()])
        return [-1, m, +1, M]

    def _plot_templates(self, bunchs, data_bounds=None):
        cluster_colors = self.cluster_color_selector.get_colors(self.cluster_ids, alpha=.75)
        for i, cluster_id in enumerate(self.cluster_ids):
            d = bunchs[cluster_id]
            wave = d.template  # shape: (n_samples, n_channels)
            channel_ids_loc = d.channel_ids
            n_channels_loc = len(channel_ids_loc)

            n_samples, nc = wave.shape
            assert nc == n_channels_loc

            # Find the x coordinates.
            t = _get_linear_x(n_channels_loc, n_samples)

            color = cluster_colors[i]
            assert len(color) == 4

            # Generate the box index (channel_idx, cluster_idx) per vertex.
            box_index = _index_of(channel_ids_loc, self.channel_ids)
            box_index = np.repeat(box_index, n_samples)
            box_index = np.c_[
                box_index.reshape((-1, 1)), i * np.ones((n_samples * n_channels_loc, 1))]
            assert box_index.shape == (n_channels_loc * n_samples, 2)
            assert box_index.size == wave.size * 2

            # Generate the waveform array.
            wave = wave.T.copy()

            self.canvas.plot_batch(
                x=t, y=wave, color=color, box_index=box_index, data_bounds=data_bounds)
        if bunchs:
            self.canvas.plot()

    def on_select(self, cluster_ids=(), **kwargs):
        if not cluster_ids:
            return
        # TODO

    def plot(self):
        # Retrieve the waveform data.
        bunchs = self.templates(self.cluster_ids)
        n_clusters = len(bunchs)
        self.canvas.grid.shape = (self.n_channels, n_clusters)

        self.canvas.clear()
        data_bounds = self._get_data_bounds(bunchs)
        self._plot_templates(bunchs, data_bounds=data_bounds)
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(TemplateView, self).attach(gui)

    def on_mouse_click(self, e):
        b = e.button
        nums = tuple('%d' % i for i in range(10))
        if 'Control' in e.modifiers or e.key in nums:
            key = int(e.key) if e.key in nums else None
            # Get mouse position in NDC.
            (channel_idx, cluster_idx), _ = self.canvas.grid.box_map(e.pos)
            cluster_id = self.cluster_ids[cluster_idx]
            logger.debug("Click on cluster %d with key %s and button %s.", cluster_id, key, b)
            emit('cluster_click', self, cluster_id, key=key, button=b)
