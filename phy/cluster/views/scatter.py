# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils._color import selected_cluster_color
from phylib.utils import connect
from .base import ManualClusteringView
from phy.plot import NDC
from phy.plot.visuals import ScatterVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(ManualClusteringView):
    _default_position = 'right'
    _default_marker_size = 5.

    def __init__(self, coords=None):
        # coords is a function cluster_ids => [Bunch(x, y) for _ in cluster_ids]
        super(ScatterView, self).__init__()
        self.canvas.enable_axes()
        self.canvas.enable_lasso()
        assert coords
        self.coords = coords
        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

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
        assert xmin <= xmax
        assert ymin <= ymax
        return (xmin, ymin, xmax, ymax)

    def _plot_points(self, bunchs, data_bounds):
        ms = self._default_marker_size
        xmin, ymin, xmax, ymax = data_bounds
        self.visual.reset_batch()
        for i, d in enumerate(bunchs):
            x, y = d.x, d.y
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape
            color = selected_cluster_color(i, .75)
            # Create one visual per cluster.
            self.visual.add_batch_data(x=x, y=y, color=color, size=ms, data_bounds=data_bounds)
        self.canvas.update_visual(self.visual)

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return

        # Retrieve the data.
        bunchs = self.coords(self.cluster_ids)
        if bunchs is None:
            return
        self.data_bounds = self._get_data_bounds(bunchs)

        self._plot_points(bunchs, self.data_bounds)
        self.canvas.axes.reset_data_bounds(self.data_bounds)
        self.canvas.update()

    def attach(self, gui):
        super(ScatterView, self).attach(gui)
        connect(self.on_request_split)

    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if (self.canvas.lasso.count < 3 or
                not len(self.cluster_ids)):  # pragma: no cover
            return np.array([], dtype=np.int64)

        # Get all points from all clusters.
        pos = []
        spike_ids = []

        bunchs = self.coords(self.cluster_ids, load_all=True)
        if bunchs is None:
            return
        assert len(bunchs) == len(self.cluster_ids)
        for cluster_id, bunch in zip(self.cluster_ids, bunchs):
            # Load all spikes.
            points = np.c_[bunch.x, bunch.y]
            pos.append(points)
            spike_ids.append(bunch.spike_ids)
        pos = np.vstack(pos)
        pos = self.visual.transforms.apply(pos)
        spike_ids = np.concatenate(spike_ids)

        # Find lassoed spikes.
        ind = self.canvas.lasso.in_polygon(pos)
        self.canvas.lasso.clear()
        return np.unique(spike_ids[ind])
