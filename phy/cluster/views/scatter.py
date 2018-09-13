# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils._color import _colormap
from .base import ManualClusteringViewMatplotlib, ManualClusteringView
from phy.plot import NDC

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class BaseScatterView(object):
    _default_marker_size = 2.

    def __init__(self,
                 coords=None,  # function clusters: Bunch(x, y)
                 **kwargs):

        assert coords
        self.coords = coords

        # Initialize the view.
        super(BaseScatterView, self).__init__(**kwargs)

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


class ScatterViewMatplotlib(BaseScatterView, ManualClusteringViewMatplotlib):
    def __init__(self, *args, **kwargs):
        super(ScatterViewMatplotlib, self).__init__(*args, **kwargs)
        self.subplots(1, 1)
        self.ax = self.axes[0, 0]

    def _plot_points(self, bunchs, data_bounds):
        ms = self._default_marker_size
        xmin, ymin, xmax, ymax = data_bounds
        for i, d in enumerate(bunchs):
            x, y = d.x, d.y
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape
            self.ax.scatter(x, y, marker=',', c=tuple(_colormap(i)) + (.5,), s=ms)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

    def on_select(self, cluster_ids=(), **kwargs):
        if not cluster_ids:
            return

        # Retrieve the data.
        bunchs = self._get_data(cluster_ids)
        data_bounds = self._get_data_bounds(bunchs)

        # Plot the points.
        self.ax.clear()
        self.config_ax(self.ax)
        self.ax.grid(color='w', alpha=.2)
        self._plot_points(bunchs, data_bounds)
        self.show()


class ScatterView(BaseScatterView, ManualClusteringView):
    def _plot_points(self, bunchs, data_bounds):
        ms = self._default_marker_size
        xmin, ymin, xmax, ymax = data_bounds
        for i, d in enumerate(bunchs):
            x, y = d.x, d.y
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape
            self.scatter(
                x=x, y=y, color=tuple(_colormap(i)) + (.5,), size=ms, data_bounds=data_bounds)

    def on_select(self, cluster_ids=(), **kwargs):
        if not cluster_ids:
            return

        # Retrieve the data.
        bunchs = self._get_data(cluster_ids)
        data_bounds = self._get_data_bounds(bunchs)

        with self.building():
            self._plot_points(bunchs, data_bounds)
