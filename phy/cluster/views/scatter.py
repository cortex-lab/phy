# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from phy.utils._color import _colormap
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(ManualClusteringView):
    _default_marker_size = 3.

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

        # Plot the points.
        with self.building():
            for i, cluster_id in enumerate(cluster_ids):
                d = self.coords(cluster_id)
                spike_ids = d.spike_ids
                x = d.x
                y = d.y
                data_bounds = d.get('data_bounds', 'auto')
                n_spikes = len(spike_ids)
                assert n_spikes > 0
                assert x.shape == (n_spikes,)
                assert y.shape == (n_spikes,)

                self.scatter(x=x, y=y,
                             color=tuple(_colormap(i)) + (.5,),
                             size=self._default_marker_size,
                             data_bounds=data_bounds,
                             )
