# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.io.array import _unique, _index_of
from phylib.utils import emit
from phylib.utils.color import _add_selected_clusters_colors

from .base import ManualClusteringView, MarkerSizeMixin
from phy.plot import NDC
from phy.plot.visuals import ScatterVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Raster view
# -----------------------------------------------------------------------------

class RasterView(MarkerSizeMixin, ManualClusteringView):
    """This view shows a raster plot of all clusters.

    Constructor
    -----------

    spike_times : array-like
        An `(n_spikes,)` array with the spike times, in seconds.
    spike_clusters : array-like
        An `(n_spikes,)` array with the spike-cluster assignments.
    cluster_ids : array-like
        The list of all clusters to show initially.
    cluster_color_selector : ClusterColorSelector
        The object managing the color mapping.

    """

    _default_position = 'right'

    default_shortcuts = {
        'increase': 'ctrl+shift++',
        'decrease': 'ctrl+shift+-',
    }

    def __init__(self, spike_times, spike_clusters, cluster_ids=None, cluster_color_selector=None):
        self.spike_times = spike_times
        self.n_spikes = len(spike_times)
        self.duration = spike_times[-1] * 1.01

        assert len(spike_clusters) == self.n_spikes
        self.set_spike_clusters(spike_clusters)
        self.set_cluster_ids(cluster_ids if cluster_ids is not None else _unique(spike_clusters))
        self.cluster_color_selector = cluster_color_selector

        super(RasterView, self).__init__()

        self.canvas.set_layout('stacked', n_plots=self.n_clusters, has_clip=False)
        self.canvas.constrain_bounds = NDC
        self.canvas.enable_axes()

        self.visual = ScatterVisual(marker='vbar')
        self.canvas.add_visual(self.visual)

    # Data-related functions
    # -------------------------------------------------------------------------

    def set_spike_clusters(self, spike_clusters):
        """Set the spike clusters for all spikes."""
        self.spike_clusters = spike_clusters

    def set_cluster_ids(self, cluster_ids):
        """Set the shown clusters, which can be filtered and in any order (from top to bottom)."""
        if len(cluster_ids) == 0:
            return
        self.cluster_ids = cluster_ids
        self.n_clusters = len(self.cluster_ids)
        # Only keep spikes that belong to the selected clusters.
        self.spike_ids = np.isin(self.spike_clusters, self.cluster_ids)

    # Internal plotting functions
    # -------------------------------------------------------------------------

    def _get_x(self):
        """Return the x position of the spikes."""
        return self.spike_times[self.spike_ids]

    def _get_y(self):
        """Return the y position of the spikes, given the relative position of the clusters."""
        return np.zeros(np.sum(self.spike_ids))

    def _get_box_index(self):
        """Return, for every spike, its row in the raster plot. This depends on the ordering
        in self.cluster_ids."""
        cl = self.spike_clusters[self.spike_ids]
        # Sanity check.
        # assert np.all(np.in1d(cl, self.cluster_ids))
        return _index_of(cl, self.cluster_ids)

    def _get_color(self, box_index, selected_clusters=None):
        """Return, for every spike, its color, based on its box index."""
        cluster_colors = self.cluster_color_selector.get_colors(self.cluster_ids, alpha=.75)
        # Selected cluster colors.
        if selected_clusters is not None:
            cluster_colors = _add_selected_clusters_colors(
                selected_clusters, self.cluster_ids, cluster_colors)
        return cluster_colors[box_index, :]

    # Main methods
    # -------------------------------------------------------------------------

    def _get_data_bounds(self):
        """Bounds of the raster plot view."""
        return (0, 0, self.duration, self.n_clusters)

    def update_cluster_sort(self, cluster_ids):
        """Update the order of all clusters."""
        self.cluster_ids = cluster_ids
        self.visual.set_box_index(self._get_box_index())
        self.canvas.update()

    def update_color(self, selected_clusters=None):
        """Update the color of the spikes, depending on the selected clustersd."""
        box_index = self._get_box_index()
        color = self._get_color(box_index, selected_clusters=selected_clusters)
        self.visual.set_color(color)
        self.canvas.update()

    def plot(self, **kwargs):
        """Make the raster plot."""
        if not len(self.spike_clusters):
            return
        x = self._get_x()  # spike times for the selected spikes
        y = self._get_y()  # just 0
        box_index = self._get_box_index()
        color = self._get_color(box_index)
        assert x.shape == y.shape == box_index.shape
        assert color.shape[0] == len(box_index)
        self.data_bounds = self._get_data_bounds()

        self.visual.set_data(
            x=x, y=y, color=color, size=self.marker_size,
            data_bounds=(0, 0, self.duration, 1))
        self.visual.set_box_index(box_index)
        self.canvas.stacked.n_boxes = self.n_clusters
        self._update_axes()
        self.canvas.update()

    def on_select(self, cluster_ids=(), **kwargs):
        """Update the view with the selected clusters."""
        if not cluster_ids:
            return
        self.update_color(selected_clusters=cluster_ids)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(RasterView, self).attach(gui)

        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.separator()

    def on_mouse_click(self, e):
        """Select a cluster by clicking in the raster plot."""
        b = e.button
        if 'Control' in e.modifiers:
            # Get mouse position in NDC.
            cluster_idx, _ = self.canvas.stacked.box_map(e.pos)
            cluster_id = self.cluster_ids[cluster_idx]
            logger.debug("Click on cluster %d with button %s.", cluster_id, b)
            emit('cluster_click', self, cluster_id, button=b)
