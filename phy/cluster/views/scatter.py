# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils.color import selected_cluster_color
from .base import ManualClusteringView, MarkerSizeMixin, LassoMixin
from phy.plot.visuals import ScatterVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(MarkerSizeMixin, LassoMixin, ManualClusteringView):
    """This view displays a scatter plot for all selected clusters.

    Constructor
    -----------

    coords : function
        Maps `cluster_ids` to a list `[Bunch(x, y, spike_ids, data_bounds), ...]` for each cluster.

    """

    _default_position = 'right'

    def __init__(self, coords=None):
        super(ScatterView, self).__init__()
        # Save the marker size in the global and local view's config.

        self.canvas.enable_axes()
        self.canvas.enable_lasso()
        assert coords
        self.coords = coords
        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

    def _plot_cluster(self, bunch):
        ms = self._marker_size
        self.visual.add_batch_data(
            pos=bunch.pos, color=bunch.color, size=ms, data_bounds=self.data_bounds)

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids."""
        bunchs = self.coords(self.cluster_ids, load_all=load_all) or ()
        # Add a pos attribute in bunchs in addition to x and y.
        for i, bunch in enumerate(bunchs):
            assert bunch.x.ndim == 1
            assert bunch.x.shape == bunch.y.shape
            bunch.pos = np.c_[bunch.x, bunch.y]
            assert bunch.pos.ndim == 2
            assert 'spike_ids' in bunch
            bunch.color = selected_cluster_color(i, .75)
        return bunchs

    def plot(self, **kwargs):
        """Update the view with the current cluster selection."""
        bunchs = self.get_clusters_data()
        # Hide the visual if there is no data.
        if not bunchs:
            self.visual.hide()
            self.canvas.update()
            return
        self.data_bounds = self._get_data_bounds(bunchs)

        self.visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.visual)
        self.visual.show()

        self._update_axes()
        self.canvas.update()
