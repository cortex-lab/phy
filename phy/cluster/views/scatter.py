# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils._color import selected_cluster_color
from .base import ManualClusteringView
from phy.plot import NDC

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(ManualClusteringView):
    _default_position = 'right'
    _default_marker_size = 5.

    def __init__(self, coords=None):  # coords is a function clusters: Bunch(x, y)
        super(ScatterView, self).__init__()
        self.canvas.enable_axes()
        assert coords
        self.coords = coords

    def _get_data(self, cluster_ids):
        return [self.coords(cluster_id) for cluster_id in cluster_ids]

    def _get_data_bounds(self, bunchs):
        if not bunchs:  # pragma: no cover
            return NDC
        xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
        for d in bunchs:
            xmin_, ymin_, xmax_, ymax_ = (
                d.get('data_bounds', None) or
                (d.x.min(), d.y.min(), d.x.max(), d.y.max()))
            xmin = min(xmin, xmin_)
            ymin = min(ymin, ymin_)
            xmax = max(xmax, xmax_)
            ymax = max(ymax, ymax_)
        return (xmin, ymin, xmax, ymax)

    def _plot_points(self, bunchs, data_bounds):
        ms = self._default_marker_size
        xmin, ymin, xmax, ymax = data_bounds
        for i, d in enumerate(bunchs):
            x, y = d.x, d.y
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape
            color = selected_cluster_color(i, .75)
            # Create one visual per cluster.
            self.canvas.scatter(x=x, y=y, color=color, size=ms, data_bounds=data_bounds)

    def on_select(self, cluster_ids=(), **kwargs):
        if not cluster_ids:
            return

        # Retrieve the data.
        bunchs = self._get_data(cluster_ids)
        data_bounds = self._get_data_bounds(bunchs)

        self.canvas.clear()
        self._plot_points(bunchs, data_bounds)
        self.canvas.axes.reset_data_bounds(data_bounds)
        self.canvas.update()
