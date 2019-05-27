# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.io.array import _unique, _index_of
from phylib.utils import emit
from phylib.utils._color import _add_selected_clusters_colors

from .base import ManualClusteringView
from phy.plot import NDC
from phy.plot.visuals import ScatterVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class RasterView(ManualClusteringView):
    _default_position = 'right'
    _marker_size = 5
    _marker_size_increment = 1.1

    default_shortcuts = {
        'increase': 'ctrl+shift++',
        'decrease': 'ctrl+shift+-',
    }

    def __init__(self, spike_times, spike_clusters, cluster_color_selector=None):
        self.spike_times = spike_times
        self.n_spikes = len(spike_times)
        self.duration = spike_times[-1] * 1.01

        assert len(spike_clusters) == self.n_spikes
        self.set_spike_clusters(spike_clusters)
        self.set_cluster_ids(_unique(spike_clusters))
        self.cluster_color_selector = cluster_color_selector

        super(RasterView, self).__init__()
        # Save the marker size in the global and local view's config.
        self.state_attrs += ('marker_size',)
        self.local_state_attrs += ('marker_size',)

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

    @property
    def data_bounds(self):
        return (0, 0, self.duration, self.n_clusters)

    def update_cluster_sort(self, cluster_ids):
        self.cluster_ids = cluster_ids
        self.visual.set_box_index(self._get_box_index())
        self.canvas.update()

    def update_color(self, selected_clusters=None):
        box_index = self._get_box_index()
        color = self._get_color(box_index, selected_clusters=selected_clusters)
        self.visual.set_color(color)
        self.canvas.update()

    def plot(self):
        if not len(self.spike_clusters):
            return
        x = self._get_x()  # spike times for the selected spikes
        y = self._get_y()  # just 0
        box_index = self._get_box_index()
        color = self._get_color(box_index)
        assert x.shape == y.shape == box_index.shape
        assert color.shape[0] == len(box_index)

        self.visual.set_data(
            x=x, y=y, color=color, size=self.marker_size,
            data_bounds=(0, 0, self.duration, 1))
        self.visual.set_box_index(box_index)
        self.canvas.stacked.n_boxes = self.n_clusters
        self.canvas.axes.reset_data_bounds(self.data_bounds, do_update=True)
        self.canvas.update()

    def on_select(self, cluster_ids=(), **kwargs):
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
        b = e.button
        if 'Control' in e.modifiers:
            # Get mouse position in NDC.
            cluster_idx, _ = self.canvas.stacked.box_map(e.pos)
            cluster_id = self.cluster_ids[cluster_idx]
            logger.debug("Click on cluster %d with button %s.", cluster_id, b)
            emit('cluster_click', self, cluster_id, button=b)

    # Marker size
    # -------------------------------------------------------------------------

    @property
    def marker_size(self):
        return self._marker_size

    @marker_size.setter
    def marker_size(self, val):
        assert val > 0
        self._marker_size = val
        self.visual.set_marker_size(val)
        self.canvas.update()

    def increase(self):
        self.marker_size *= self._marker_size_increment

    def decrease(self):
        self.marker_size = max(.1, self.marker_size / self._marker_size_increment)
