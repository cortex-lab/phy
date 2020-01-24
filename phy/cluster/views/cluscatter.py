# -*- coding: utf-8 -*-

"""Cluster scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils.color import _add_selected_clusters_colors
from phylib.utils import emit

from phy.plot.transform import range_transform, NDC
from phy.plot.visuals import ScatterVisual
from .base import ManualClusteringView, BaseGlobalView, MarkerSizeMixin, BaseColorView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Template view
# -----------------------------------------------------------------------------

class ClusterScatterView(MarkerSizeMixin, BaseColorView, BaseGlobalView, ManualClusteringView):
    """This view shows all clusters in a customizable scatter plot.

    Constructor
    -----------

    cluster_ids : array-like
    cluster_info: function
        Maps cluster_id => Bunch() with attributes.
    bindings: dict
        Maps plot dimension to cluster attributes.

    """
    _default_position = 'right'
    _scaling = 1.
    _default_alpha = .75
    _min_marker_size = 5.0
    _max_marker_size = 30.0

    # NOTE: this is not the actual marker size, but a scaling factor for the normal marker size.
    _marker_size = 1.
    _default_marker_size = 1.

    logarithmic_size = False

    default_shortcuts = {
    }

    def __init__(
            self, cluster_ids=None, cluster_info=None, bindings=None, **kwargs):
        super(ClusterScatterView, self).__init__(**kwargs)
        self.state_attrs += ()
        self.local_state_attrs += ('scaling',)

        self.canvas.enable_axes()
        self.cluster_info = cluster_info
        self.bindings = bindings
        assert set(('x_axis', 'y_axis', 'size')) <= set(bindings.keys())

        # Full list of clusters.
        if cluster_ids is not None:
            self.set_cluster_ids(cluster_ids)

        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

    # Data access
    # -------------------------------------------------------------------------

    def get_cluster_data(self, cluster_id):
        """Return the data of one cluster."""
        data = self.cluster_info(cluster_id)
        return {k: data[v] for k, v in self.bindings.items()}

    def get_clusters_data(self, cluster_ids):
        """Return the data of a set of clusters, as a dictionary {cluster_id: Bunch}."""
        return {cluster_id: self.get_cluster_data(cluster_id) for cluster_id in cluster_ids}

    def set_cluster_ids(self, all_cluster_ids):
        """Update the cluster data by specifying the list of all cluster ids."""
        self.all_cluster_ids = all_cluster_ids
        if len(all_cluster_ids) == 0:
            return
        self.prepare_data()

    # Data preparation
    # -------------------------------------------------------------------------

    def prepare_data(self):
        """Prepare the marker position, size, and color from the cluster information."""
        self.prepare_position()
        self.prepare_size()
        self.prepare_color()

    def prepare_position(self):
        """Compute the marker positions."""
        self.cluster_data = self.get_clusters_data(self.all_cluster_ids)

        # Get the list of fields returned by cluster_info.
        self.fields = sorted(self.cluster_info(self.all_cluster_ids[0]).keys())

        # Create the x, y, size, colors arrays.
        x = np.array(
            [self.cluster_data[cluster_id]['x_axis'] for cluster_id in self.all_cluster_ids])
        y = np.array(
            [self.cluster_data[cluster_id]['y_axis'] for cluster_id in self.all_cluster_ids])

        self.marker_positions = np.c_[x, y]

        # Update the data bounds.
        self.data_bounds = (x.min(), y.min(), x.max(), y.max())

    def prepare_size(self):
        """Compute the marker sizes."""
        size = np.array(
            [self.cluster_data[cluster_id]['size'] for cluster_id in self.all_cluster_ids])

        # Normalize the marker size.
        if self.logarithmic_size:
            size = np.log(1.0 + size - size.min())
        m, M = size.min(), size.max()
        size = (size - m) / (M - m)  # size is in [0, 1]
        ms, Ms = self._min_marker_size, self._max_marker_size
        size = ms + size * (Ms - ms)  # now, size is in [ms, Ms]
        self.marker_sizes = size

    def prepare_color(self):
        """Compute the marker colors."""
        colors = self.get_cluster_colors(self.all_cluster_ids, self._default_alpha)
        selected_clusters = self.cluster_ids
        if selected_clusters is not None and len(selected_clusters) > 0:
            colors = _add_selected_clusters_colors(selected_clusters, self.all_cluster_ids, colors)
        self.marker_colors = colors

    # Plotting functions
    # -------------------------------------------------------------------------

    @property
    def marker_size(self):
        """Size of the spike markers, in pixels."""
        return self._marker_size

    @marker_size.setter
    def marker_size(self, val):
        # We override this method so as to use self._marker_size as a scaling factor, not
        # as an actual fixed marker size.
        self._marker_size = val
        self._set_marker_size()
        self.canvas.update()

    def _set_marker_size(self):
        self.visual.set_marker_size(self.marker_sizes * self._marker_size)

    def update_color(self):
        """Update the cluster colors depending on the selected clusters. To be overriden."""
        self.prepare_color()
        self.visual.set_color(self.marker_colors)
        self.canvas.update()

    def plot(self, **kwargs):
        """Make the scatter plot."""
        self.visual.set_data(
            pos=self.marker_positions, color=self.marker_colors,
            size=self.marker_sizes * self._marker_size,  # marker size scaling factor
            data_bounds=self.data_bounds)
        self.canvas.axes.reset_data_bounds(self.data_bounds)
        self.canvas.update()

    def change_bindings(self, **kwargs):
        """Change the bindings."""
        self.bindings.update(kwargs)
        self.prepare_data()
        self.plot()

    def attach(self, gui):
        """Attach the GUI."""
        super(ClusterScatterView, self).attach(gui)

        def _make_action(dim, name):
            def callback():
                self.change_bindings(**{dim: name})
            return callback

        # Change the bindings.
        for dim in ('x_axis', 'y_axis', 'size'):
            gui.get_submenu(self.name, 'Change %s' % dim)
            for name in self.fields:
                self.actions.add(
                    _make_action(dim, name),
                    name="Change %s to %s" % (dim, name), submenu='Change %s' % dim)

        # Toggle logarithmic size.
        self.actions.add(
            self.toggle_logarithmic_size, checkable=True, checked=self.logarithmic_size)

        self.actions.separator()

    def toggle_logarithmic_size(self, checked):
        """Toggle logarithmic scaling for marker size."""
        self.logarithmic_size = checked
        self.prepare_size()
        self._set_marker_size()
        self.canvas.update()

    def on_mouse_click(self, e):
        """Select a cluster by clicking on its template waveform."""
        b = e.button
        pos = self.canvas.window_to_ndc(e.pos)
        pos = range_transform([NDC], [self.data_bounds], [pos])[0]
        cluster_rel = np.argmin(((self.marker_positions - pos) ** 2).sum(axis=1))
        cluster_id = self.all_cluster_ids[cluster_rel]
        logger.debug("Click on cluster %d with button %s.", cluster_id, b)
        if 'Shift' in e.modifiers:
            emit('select_more', self, [cluster_id])
        else:
            emit('request_select', self, [cluster_id])
