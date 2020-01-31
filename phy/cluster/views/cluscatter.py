# -*- coding: utf-8 -*-

"""Cluster scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils.color import _add_selected_clusters_colors
from phylib.utils import emit, connect, unconnect

from phy.plot.transform import range_transform, NDC
from phy.plot.visuals import ScatterVisual, TextVisual
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
    _dims = ('x_axis', 'y_axis', 'size')

    # NOTE: this is not the actual marker size, but a scaling factor for the normal marker size.
    _marker_size = 1.
    _default_marker_size = 1.

    x_axis = ''
    y_axis = ''
    size = ''
    x_axis_log_scale = False
    y_axis_log_scale = False
    size_log_scale = False

    default_shortcuts = {
        'change_marker_size': 'alt+wheel',
        'switch_color_scheme': 'shift+wheel',
        'select_cluster': 'click',
        'select_more': 'shift+click',
        'add_to_lasso': 'control+left click',
        'clear_lasso': 'control+right click',
    }

    default_snippets = {
        'set_x_axis': 'csx',
        'set_y_axis': 'csy',
        'set_size': 'css',
    }

    def __init__(
            self, cluster_ids=None, cluster_info=None, bindings=None, **kwargs):
        super(ClusterScatterView, self).__init__(**kwargs)
        self.state_attrs += (
            'scaling',
            'x_axis', 'y_axis', 'size',
            'x_axis_log_scale', 'y_axis_log_scale', 'size_log_scale',
        )
        self.local_state_attrs += ()

        self.canvas.enable_axes()
        self.canvas.enable_lasso()

        bindings = bindings or {}
        self.cluster_info = cluster_info
        # update self.x_axis, y_axis, size
        self.__dict__.update({(k, v) for k, v in bindings.items() if k in self._dims})

        # Size range computed initially so that it doesn't change during the course of the session.
        self._size_min = self._size_max = None

        # Full list of clusters.
        self.all_cluster_ids = cluster_ids

        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

        self.label_visual = TextVisual()
        self.canvas.add_visual(self.label_visual, exclude_origins=(self.canvas.panzoom,))

        self.marker_positions = self.marker_colors = self.marker_sizes = None

    def _update_labels(self):
        self.label_visual.set_data(
            pos=[[-1, -1], [1, 1]], text=[self.x_axis, self.y_axis],
            anchor=[[1.25, 3], [-3, -1.25]])

    # Data access
    # -------------------------------------------------------------------------

    @property
    def bindings(self):
        return {k: getattr(self, k) for k in self._dims}

    def get_cluster_data(self, cluster_id):
        """Return the data of one cluster."""
        data = self.cluster_info(cluster_id)
        return {k: data.get(v, 0.) for k, v in self.bindings.items()}

    def get_clusters_data(self, cluster_ids):
        """Return the data of a set of clusters, as a dictionary {cluster_id: Bunch}."""
        return {cluster_id: self.get_cluster_data(cluster_id) for cluster_id in cluster_ids}

    def set_cluster_ids(self, all_cluster_ids):
        """Update the cluster data by specifying the list of all cluster ids."""
        self.all_cluster_ids = all_cluster_ids
        if len(all_cluster_ids) == 0:
            return
        self.prepare_position()
        self.prepare_size()
        self.prepare_color()

    # Data preparation
    # -------------------------------------------------------------------------

    def set_fields(self):
        data = self.cluster_info(self.all_cluster_ids[0])
        self.fields = sorted(data.keys())
        self.fields = [f for f in self.fields if not isinstance(data[f], str)]

    def prepare_data(self):
        """Prepare the marker position, size, and color from the cluster information."""
        self.prepare_position()
        self.prepare_size()
        self.prepare_color()

    def prepare_position(self):
        """Compute the marker positions."""
        self.cluster_data = self.get_clusters_data(self.all_cluster_ids)

        # Get the list of fields returned by cluster_info.
        self.set_fields()

        # Create the x array.
        x = np.array(
            [self.cluster_data[cluster_id]['x_axis'] or 0. for cluster_id in self.all_cluster_ids])
        if self.x_axis_log_scale:
            x = np.log(1.0 + x - x.min())

        # Create the y array.
        y = np.array(
            [self.cluster_data[cluster_id]['y_axis'] or 0. for cluster_id in self.all_cluster_ids])
        if self.y_axis_log_scale:
            y = np.log(1.0 + y - y.min())

        self.marker_positions = np.c_[x, y]

        # Update the data bounds.
        self.data_bounds = (x.min(), y.min(), x.max(), y.max())

    def prepare_size(self):
        """Compute the marker sizes."""
        size = np.array(
            [self.cluster_data[cluster_id]['size'] or 1. for cluster_id in self.all_cluster_ids])
        # Log scale for the size.
        if self.size_log_scale:
            size = np.log(1.0 + size - size.min())
        # Find the size range.
        if self._size_min is None:
            self._size_min, self._size_max = size.min(), size.max()
        m, M = self._size_min, self._size_max
        # Normalize the marker size.
        size = (size - m) / ((M - m) or 1.0)  # size is in [0, 1]
        ms, Ms = self._min_marker_size, self._max_marker_size
        size = ms + size * (Ms - ms)  # now, size is in [ms, Ms]
        self.marker_sizes = size

    def prepare_color(self):
        """Compute the marker colors."""
        colors = self.get_cluster_colors(self.all_cluster_ids, self._default_alpha)
        self.marker_colors = colors

    # Marker size
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
        if self.marker_sizes is not None:
            self.visual.set_marker_size(self.marker_sizes * self._marker_size)

    # Plotting functions
    # -------------------------------------------------------------------------

    def update_color(self):
        """Update the cluster colors depending on the current color scheme."""
        self.prepare_color()
        self.visual.set_color(self.marker_colors)
        self.canvas.update()

    def update_select_color(self):
        """Update the cluster colors after the cluster selection changes."""
        if self.marker_colors is None:
            return
        selected_clusters = self.cluster_ids
        if selected_clusters is not None and len(selected_clusters) > 0:
            colors = _add_selected_clusters_colors(
                selected_clusters, self.all_cluster_ids, self.marker_colors.copy())
            self.visual.set_color(colors)
            self.canvas.update()

    def plot(self, **kwargs):
        """Make the scatter plot."""
        if self.marker_positions is None:
            self.prepare_data()
        self.visual.set_data(
            pos=self.marker_positions, color=self.marker_colors,
            size=self.marker_sizes * self._marker_size,  # marker size scaling factor
            data_bounds=self.data_bounds)
        self.canvas.axes.reset_data_bounds(self.data_bounds)
        self.canvas.update()

    def change_bindings(self, **kwargs):
        """Change the bindings."""
        # Ensure the specified fields are valid.
        kwargs = {k: v for k, v in kwargs.items() if v in self.fields}
        assert set(kwargs.keys()) <= set(self._dims)
        # Reset the size scaling.
        if 'size' in kwargs:
            self._size_min = self._size_max = None
        self.__dict__.update(kwargs)
        self._update_labels()
        self.update_status()
        self.prepare_data()
        self.plot()

    def toggle_log_scale(self, dim, checked):
        """Toggle logarithmic scaling for one of the dimensions."""
        self._size_min = None
        setattr(self, '%s_log_scale' % dim, checked)
        self.prepare_data()
        self.plot()
        self.canvas.update()

    def set_x_axis(self, field):
        """Set the dimension for the x axis."""
        self.change_bindings(x_axis=field)

    def set_y_axis(self, field):
        """Set the dimension for the y axis."""
        self.change_bindings(y_axis=field)

    def set_size(self, field):
        """Set the dimension for the marker size."""
        self.change_bindings(size=field)

    # Misc functions
    # -------------------------------------------------------------------------

    def attach(self, gui):
        """Attach the GUI."""
        super(ClusterScatterView, self).attach(gui)

        def _make_action(dim, name):
            def callback():
                self.change_bindings(**{dim: name})
            return callback

        def _make_log_toggle(dim):
            def callback(checked):
                self.toggle_log_scale(dim, checked)
            return callback

        # Change the bindings.
        for dim in self._dims:
            view_submenu = 'Change %s' % dim

            # Change to every cluster info.
            for name in self.fields:
                self.actions.add(
                    _make_action(dim, name), show_shortcut=False,
                    name='Change %s to %s' % (dim, name), view_submenu=view_submenu)

            # Toggle logarithmic scale.
            self.actions.separator(view_submenu=view_submenu)
            self.actions.add(
                _make_log_toggle(dim), checkable=True, view_submenu=view_submenu,
                name='Toggle log scale for %s' % dim, show_shortcut=False,
                checked=getattr(self, '%s_log_scale' % dim))

        self.actions.separator()
        self.actions.add(self.set_x_axis, prompt=True, prompt_default=lambda: self.x_axis)
        self.actions.add(self.set_y_axis, prompt=True, prompt_default=lambda: self.y_axis)
        self.actions.add(self.set_size, prompt=True, prompt_default=lambda: self.size)

        connect(self.on_select)
        connect(self.on_cluster)

        @connect(sender=self.canvas)
        def on_lasso_updated(sender, polygon):
            if len(polygon) < 3:
                return
            pos = range_transform([self.data_bounds], [NDC], self.marker_positions)
            ind = self.canvas.lasso.in_polygon(pos)
            cluster_ids = self.all_cluster_ids[ind]
            emit("request_select", self, list(cluster_ids))

        @connect(sender=self)
        def on_close_view(view_, gui):
            """Unconnect all events when closing the view."""
            unconnect(self.on_select)
            unconnect(self.on_cluster)
            unconnect(on_lasso_updated)

        if self.all_cluster_ids is not None:
            self.set_cluster_ids(self.all_cluster_ids)
        self._update_labels()

    def on_select(self, *args, **kwargs):
        super(ClusterScatterView, self).on_select(*args, **kwargs)
        self.update_select_color()

    def on_cluster(self, sender, up):
        if 'all_cluster_ids' in up:
            self.set_cluster_ids(up.all_cluster_ids)
            self.plot()

    @property
    def status(self):
        return 'Size: %s. Color scheme: %s.' % (self.size, self.color_scheme)

    # Interactivity
    # -------------------------------------------------------------------------

    def on_mouse_click(self, e):
        """Select a cluster by clicking on its template waveform."""
        if 'Control' in e.modifiers:
            return
        b = e.button
        pos = self.canvas.window_to_ndc(e.pos)
        marker_pos = range_transform([self.data_bounds], [NDC], self.marker_positions)
        cluster_rel = np.argmin(((marker_pos - pos) ** 2).sum(axis=1))
        cluster_id = self.all_cluster_ids[cluster_rel]
        logger.debug("Click on cluster %d with button %s.", cluster_id, b)
        if 'Shift' in e.modifiers:
            emit('select_more', self, [cluster_id])
        else:
            emit('request_select', self, [cluster_id])
