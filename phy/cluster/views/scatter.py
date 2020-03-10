# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import inspect
import logging

import numpy as np

from phy.utils.color import selected_cluster_color, spike_colors
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

    # Do not show too many clusters.
    max_n_clusters = 8

    _default_position = 'right'

    default_shortcuts = {
        'change_marker_size': 'alt+wheel',
    }

    def __init__(self, coords=None, **kwargs):
        super(ScatterView, self).__init__(**kwargs)
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

    def _get_split_cluster_data(self, bunchs):
        """Get the data when there is one Bunch per cluster."""
        # Add a pos attribute in bunchs in addition to x and y.
        for i, (cluster_id, bunch) in enumerate(zip(self.cluster_ids, bunchs)):
            bunch.cluster_id = cluster_id
            if 'pos' not in bunch:
                assert bunch.x.ndim == 1
                assert bunch.x.shape == bunch.y.shape
                bunch.pos = np.c_[bunch.x, bunch.y]
            assert bunch.pos.ndim == 2
            assert 'spike_ids' in bunch
            bunch.color = selected_cluster_color(i, .75)
        return bunchs

    def _get_collated_cluster_data(self, bunch):
        """Get the data when there is a single Bunch for all selected clusters."""
        assert 'spike_ids' in bunch
        if 'pos' not in bunch:
            assert bunch.x.ndim == 1
            assert bunch.x.shape == bunch.y.shape
            bunch.pos = np.c_[bunch.x, bunch.y]
        assert bunch.pos.ndim == 2
        bunch.color = spike_colors(bunch.spike_clusters, self.cluster_ids)
        return bunch

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids."""
        if not load_all:
            bunchs = self.coords(self.cluster_ids) or ()
        elif 'load_all' in inspect.signature(self.coords).parameters:
            bunchs = self.coords(self.cluster_ids, load_all=load_all) or ()
        else:
            logger.warning(
                "The view `%s` may not load all spikes when using the lasso for splitting.",
                self.__class__.__name__)
            bunchs = self.coords(self.cluster_ids)
        if isinstance(bunchs, dict):
            return [self._get_collated_cluster_data(bunchs)]
        elif isinstance(bunchs, (list, tuple)):
            return self._get_split_cluster_data(bunchs)
        raise ValueError("The output of `coords()` should be either a list of Bunch, or a Bunch.")

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
