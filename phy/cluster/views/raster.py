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
    marker_size = 10

    def __init__(self, spike_times, spike_clusters, cluster_color_selector=None):
        self.spike_times = spike_times
        self.n_spikes = len(spike_times)
        self.duration = spike_times[-1] * 1.01

        assert len(spike_clusters) == self.n_spikes
        self.set_spike_clusters(spike_clusters)
        self.set_cluster_ids(_unique(spike_clusters))
        self.cluster_color_selector = cluster_color_selector

        super(RasterView, self).__init__()
        self.canvas.constrain_bounds = NDC
        self.canvas.enable_axes(data_bounds=self.data_bounds)

    def set_spike_clusters(self, spike_clusters):
        """Set the spike clusters for all spikes."""
        self.spike_clusters = spike_clusters
        self.data_bounds = (0, 0, self.duration, spike_clusters.max() + 1)

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
        return _index_of(self.spike_clusters[self.spike_ids], self.cluster_ids)

    def _get_color(self, spike_clusters_rel):
        cluster_colors = self.cluster_color_selector.get_colors(self.cluster_ids)
        cluster_colors[:, 3] = .75  # alpha channel
        return cluster_colors[spike_clusters_rel, :]

    def plot(self):
        n = len(self.cluster_ids)
        x = self._get_x()  # spike times for the selected spikes
        y = self._get_y()  # relative cluster index, in the specified cluster order
        color = self._get_color(y)

        # NOTE: minus because we count from top to bottom.
        self.canvas.clear()
        self.canvas.scatter(
            x=x, y=n - 1 - y, color=color, size=self.marker_size, marker='vbar',
            data_bounds=self.data_bounds)
        self.canvas.axes.reset_data_bounds(self.data_bounds)
        self.canvas.update()

    def on_select(self, cluster_ids=(), **kwargs):
        pass
        # TODO: change color
