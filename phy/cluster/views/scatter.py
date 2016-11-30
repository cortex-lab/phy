# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils._color import _colormap
from .base import ManualClusteringView

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

    def on_select(self, cluster_ids=None):
        super(ScatterView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Retrieve the data.
        ds = [self.coords(cluster_id) for cluster_id in cluster_ids]

        # Compute the data bounds.
        data_bounds = ds[0].get('data_bounds', None)
        if data_bounds is None:
            xmin = np.min([d.x.min() for d in ds])
            ymin = np.min([d.y.min() for d in ds])
            xmax = np.max([d.x.max() for d in ds])
            ymax = np.max([d.y.max() for d in ds])
            data_bounds = (xmin, ymin, xmax, ymax)

        # Plot the points.
        with self.building():
            for i, cluster_id in enumerate(ds):
                d = ds[i]
                x = d.x
                y = d.y
                assert x.ndim == y.ndim == 1
                assert x.shape == y.shape

                self.scatter(x=x, y=y,
                             color=tuple(_colormap(i)) + (.5,),
                             size=self._default_marker_size,
                             data_bounds=data_bounds,
                             )
