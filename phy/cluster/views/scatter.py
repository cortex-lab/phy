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
        for i, d in enumerate(bunchs):
            x, y = d.x, d.y
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape

            self.scatter(x=x, y=y,
                         color=tuple(_colormap(i)) + (.5,),
                         size=self._default_marker_size,
                         data_bounds=data_bounds,
                         )

    def on_select(self, cluster_ids=None, **kwargs):
        super(ScatterView, self).on_select(cluster_ids, **kwargs)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Retrieve the data.
        bunchs = self._get_data(cluster_ids)

        # Compute the data bounds.
        data_bounds = self._get_data_bounds(bunchs)

        # Plot the points.
        with self.building():
            self._plot_points(bunchs, data_bounds)
