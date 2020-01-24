# -*- coding: utf-8 -*-

"""Cluster scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils.color import _add_selected_clusters_colors
from phylib.io.array import _index_of
from phylib.utils import emit, Bunch, connect

from phy.plot import get_linear_x
from phy.plot.visuals import ScatterVisual
from .base import ManualClusteringView, BaseGlobalView, ScalingMixin, BaseColorView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Template view
# -----------------------------------------------------------------------------

class ClusterScatterView(ScalingMixin, BaseColorView, BaseGlobalView, ManualClusteringView):
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

    default_shortcuts = {
        'next_color_scheme': 'b'
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

        connect(self.on_color_scheme_changed, sender=self)

    # Internal plot functions
    # -------------------------------------------------------------------------

    def get_cluster_data(self, cluster_id):
        data = self.cluster_info(cluster_id)
        return {k: data[v] for k, v in self.bindings.items()}

    def get_clusters_data(self, cluster_ids):
        return {cluster_id: self.get_cluster_data(cluster_id) for cluster_id in cluster_ids}

    def set_cluster_ids(self, cluster_ids):
        self.all_cluster_ids = cluster_ids
        data = self.get_clusters_data(cluster_ids)
        x = np.array([data[cluster_id]['x_axis'] for cluster_id in cluster_ids])
        y = np.array([data[cluster_id]['y_axis'] for cluster_id in cluster_ids])
        size = np.array([data[cluster_id]['size'] for cluster_id in cluster_ids])
        colors = self.get_cluster_colors(cluster_ids, self._default_alpha)
        self.data_bounds = (x.min(), y.min(), x.max(), y.max())
        self.plot_data = Bunch(x=x, y=y, size=size, color=colors, data_bounds=self.data_bounds)

    def on_color_scheme_changed(self, sender, name):
        # TODO: default handler in base class with self.cluster_ids??
        self.update_color(self.cluster_ids)

    def update_color(self, selected_clusters=None):
        """Update the cluster colors depending on the selected clusters. To be overriden."""
        colors = self.get_cluster_colors(self.all_cluster_ids, self._default_alpha)
        if selected_clusters is not None and len(selected_clusters) > 0:
            colors = _add_selected_clusters_colors(selected_clusters, self.all_cluster_ids, colors)
        self.visual.set_color(colors)
        self.canvas.update()

    def plot(self, **kwargs):
        """Make the scatter plot."""
        self.visual.set_data(**self.plot_data)
        self.canvas.axes.reset_data_bounds(self.data_bounds)
        self.canvas.update()

    # def on_mouse_click(self, e):
    #     """Select a cluster by clicking on its template waveform."""
    #     b = e.button
    #     if 'Control' in e.modifiers or 'Shift' in e.modifiers:
    #         logger.debug("Click on cluster %d with button %s.", cluster_id, b)
    #         if 'Shift' in e.modifiers:
    #             emit('select_more', self, [cluster_id])
    #         else:
    #             emit('request_select', self, [cluster_id])

    # Scaling
    # -------------------------------------------------------------------------

    def _set_scaling_value(self, value):
        self._scaling = value

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value
