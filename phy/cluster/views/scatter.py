# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils._color import _colormap
from .base import ManualClusteringView
from phy.plot import NDC

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(ManualClusteringView):
    _default_marker_size = 5.

    def __init__(self,
                 coords=None,  # function clusters: Bunch(x, y)
                 **kwargs):
        assert coords
        self.coords = coords

        # Initialize the view.
        super(ScatterView, self).__init__(**kwargs)

    def _get_data(self, cluster_ids):
        return [self.coords(cluster_id) for cluster_id in cluster_ids]

    def _get_data_bounds(self, bunchs):
        if not bunchs:  # pragma: no cover
            return NDC
        data_bounds = bunchs[0].get('data_bounds', None)
        if data_bounds is None:
            xmin = np.min([d.x.min() for d in bunchs])
            ymin = np.min([d.y.min() for d in bunchs])
            xmax = np.max([d.x.max() for d in bunchs])
            ymax = np.max([d.y.max() for d in bunchs])
            data_bounds = (xmin, ymin, xmax, ymax)
        return data_bounds

    def _plot_points(self, bunchs, data_bounds):
        ms = self._default_marker_size
        xmin, ymin, xmax, ymax = data_bounds
        for i, d in enumerate(bunchs):
            x, y = d.x, d.y
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape
            color = _colormap(i, .75)
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
        self.canvas.enable_axes(data_bounds=data_bounds)
