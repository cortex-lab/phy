# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.io.array import _index_of
from phylib.utils import emit
from phy.utils.color import _add_selected_clusters_colors

from .base import ManualClusteringView, BaseGlobalView, MarkerSizeMixin, BaseColorView
from phy.plot.visuals import ScatterVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Raster view
# -----------------------------------------------------------------------------

class RasterView(MarkerSizeMixin, BaseColorView, BaseGlobalView, ManualClusteringView):
    """This view shows a raster plot of all clusters.

    Constructor
    -----------

    spike_times : array-like
        An `(n_spikes,)` array with the spike times, in seconds.
    spike_clusters : array-like
        An `(n_spikes,)` array with the spike-cluster assignments.
    cluster_ids : array-like
        The list of all clusters to show initially.

    """

    _default_position = 'right'

    default_shortcuts = {
        'change_marker_size': 'alt+wheel',
        'switch_color_scheme': 'shift+wheel',
        'decrease_marker_size': 'ctrl+shift+-',
        'increase_marker_size': 'ctrl+shift++',
        'select_cluster': 'ctrl+click',
        'select_more': 'shift+click',
    }

    def __init__(self, spike_times, spike_clusters, cluster_ids=None, **kwargs):
        self.spike_times = spike_times
        self.n_spikes = len(spike_times)
        self.duration = spike_times[-1] * 1.01
        self.n_clusters = 1

        assert len(spike_clusters) == self.n_spikes
        self.set_spike_clusters(spike_clusters)
        self.set_cluster_ids(cluster_ids)

        super(RasterView, self).__init__(**kwargs)

        self.canvas.set_layout('stacked', origin='top', n_plots=self.n_clusters, has_clip=False)
        self.canvas.enable_axes()

        self.visual = ScatterVisual(
            marker='vbar',
            marker_scaling='''
                point_size = v_size * u_zoom.y + 5.;
                float width = 0.2;
                float height = 0.5;
                vec2 marker_size = point_size * vec2(width, height);
                marker_size.x = clamp(marker_size.x, 1, 20);
            ''',
        )
        self.visual.inserter.insert_vert('''
                gl_PointSize = a_size * u_zoom.y + 5.0;
        ''', 'end')
        self.canvas.add_visual(self.visual)
        self.canvas.panzoom.set_constrain_bounds((-1, -2, +1, +2))

    # Data-related functions
    # -------------------------------------------------------------------------

    def set_spike_clusters(self, spike_clusters):
        """Set the spike clusters for all spikes."""
        self.spike_clusters = spike_clusters

    def set_cluster_ids(self, cluster_ids):
        """Set the shown clusters, which can be filtered and in any order (from top to bottom)."""
        if cluster_ids is None or not len(cluster_ids):
            return
        self.all_cluster_ids = cluster_ids
        self.n_clusters = len(self.all_cluster_ids)
        # Only keep spikes that belong to the selected clusters.
        self.spike_ids = np.isin(self.spike_clusters, self.all_cluster_ids)

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
        return _index_of(cl, self.all_cluster_ids)

    def _get_color(self, box_index, selected_clusters=None):
        """Return, for every spike, its color, based on its box index."""
        cluster_colors = self.get_cluster_colors(self.all_cluster_ids, alpha=.75)
        # Selected cluster colors.
        if selected_clusters is not None:
            cluster_colors = _add_selected_clusters_colors(
                selected_clusters, self.all_cluster_ids, cluster_colors)
        return cluster_colors[box_index, :]

    # Main methods
    # -------------------------------------------------------------------------

    def _get_data_bounds(self):
        """Bounds of the raster plot view."""
        return (0, 0, self.duration, self.n_clusters)

    def update_cluster_sort(self, cluster_ids):
        """Update the order of all clusters."""
        self.all_cluster_ids = cluster_ids
        self.visual.set_box_index(self._get_box_index())
        self.canvas.update()

    def update_color(self):
        """Update the color of the spikes, depending on the selected clusters."""
        box_index = self._get_box_index()
        color = self._get_color(box_index, selected_clusters=self.cluster_ids)
        self.visual.set_color(color)
        self.canvas.update()

    @property
    def status(self):
        return 'Color scheme: %s' % self.color_scheme

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
            data_bounds=(0, -1, self.duration, 1))
        self.visual.set_box_index(box_index)
        self.canvas.stacked.n_boxes = self.n_clusters
        self._update_axes()
        # self.canvas.stacked.add_boxes(self.canvas)
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(RasterView, self).attach(gui)

        self.actions.add(self.increase_marker_size)
        self.actions.add(self.decrease_marker_size)
        self.actions.separator()

    def on_select(self, *args, **kwargs):
        super(RasterView, self).on_select(*args, **kwargs)
        self.update_color()

    def zoom_to_time_range(self, interval):
        """Zoom to a time interval."""
        if not interval:
            return
        t0, t1 = interval
        w = .5 * (t1 - t0)  # half width
        tm = .5 * (t0 + t1)
        w = min(5, w)  # minimum 5s time range
        t0, t1 = tm - w, tm + w
        x0 = -1 + 2 * t0 / self.duration
        x1 = -1 + 2 * t1 / self.duration
        box = (x0, -1, x1, +1)
        self.canvas.panzoom.set_range(box)

    # Interactivity
    # -------------------------------------------------------------------------

    def on_mouse_click(self, e):
        """Select a cluster by clicking in the raster plot."""
        if 'Control' not in e.modifiers:
            return
        b = e.button
        # Get mouse position in NDC.
        cluster_idx, _ = self.canvas.stacked.box_map(e.pos)
        cluster_id = self.all_cluster_ids[cluster_idx]
        logger.debug("Click on cluster %d with button %s.", cluster_id, b)
        if 'Shift' in e.modifiers:
            emit('select_more', self, [cluster_id])
        else:
            emit('request_select', self, [cluster_id])
