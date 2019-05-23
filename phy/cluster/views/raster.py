# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.io.array import _unique, _index_of

from .base import ManualClusteringView
from phy.plot import NDC

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class RasterView(ManualClusteringView):
    _default_position = 'right'
    marker_size = 5

    def __init__(self, spike_times, spike_clusters, cluster_color_selector=None):
        self.spike_times = spike_times
        self.n_spikes = len(spike_times)
        self.duration = spike_times[-1] * 1.01

        assert len(spike_clusters) == self.n_spikes
        self.set_spike_clusters(spike_clusters)
        self.set_cluster_ids(_unique(spike_clusters))
        self.cluster_color_selector = cluster_color_selector

        super(RasterView, self).__init__()
        self.canvas.set_layout('stacked', n_plots=self.n_clusters, has_clip=False)
        self.canvas.constrain_bounds = NDC
        self.canvas.enable_axes(show_y=False)

    def set_spike_clusters(self, spike_clusters):
        """Set the spike clusters for all spikes."""
        self.spike_clusters = spike_clusters

    def set_cluster_ids(self, cluster_ids):
        """Set the shown clusters, which can be filtered and in any order (from top to bottom)."""
        self.cluster_ids = cluster_ids
        self.n_clusters = len(self.cluster_ids)
        # Only keep spikes that belong to the selected clusters.
        self.spike_ids = np.in1d(self.spike_clusters, self.cluster_ids)

    def _get_x(self):
        return self.spike_times[self.spike_ids]

    def _get_y(self):
        """Return the y position of the spikes, given the relative position of the clusters."""
        return np.zeros_like(self.spike_ids)

    def _get_box_index(self):
        """Return, for every spike, its row in the raster plot. This depends on the ordering
        in self.cluster_ids."""
        cl = self.spike_clusters[self.spike_ids]
        # Sanity check.
        assert np.all(np.in1d(cl, self.cluster_ids))
        return _index_of(cl, self.cluster_ids)

    def _get_color(self, box_index):
        """Return, for every spike, its color, based on its box index."""
        cluster_colors = self.cluster_color_selector.get_colors(self.cluster_ids, alpha=.75)
        return cluster_colors[box_index, :]

    def plot(self):
        if not len(self.spike_clusters):
            return
        x = self._get_x()  # spike times for the selected spikes
        y = self._get_y()  # just 0
        box_index = self._get_box_index()
        color = self._get_color(box_index)
        # ymax = y.max()
        data_bounds = (0, 0, self.duration, self.n_clusters)

        self.canvas.clear()
        self.canvas.scatter(
            x=x, y=y, color=color, size=self.marker_size, marker='vbar',
            box_index=box_index, data_bounds=(0, 0, self.duration, 1))
        self.canvas.stacked.n_plots = self.n_clusters
        self.canvas.axes.reset_data_bounds(data_bounds, do_update=True)
        self.canvas.update()

    def on_select(self, cluster_ids=(), **kwargs):
        # self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        # TODO
