# -*- coding: utf-8 -*-

"""Scatter view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils.geometry import range_transform
from phylib.utils.color import selected_cluster_color
from phylib.utils import connect
from .base import ManualClusteringView
from phy.plot import NDC
from phy.plot.visuals import ScatterVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(ManualClusteringView):
    """This view displays a scatter plot for all selected clusters.

    Constructor
    -----------

    coords : function
        Maps `cluster_ids` to a list `[Bunch(x, y), ...]` for each cluster.

    """

    _default_position = 'right'
    _marker_size = 5.
    _marker_size_increment = 1.1

    def __init__(self, coords=None):
        super(ScatterView, self).__init__()
        # Save the marker size in the global and local view's config.
        self.state_attrs += ('marker_size',)
        self.local_state_attrs += ('marker_size',)

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

        if xmin == xmax == 0:
            xmin, xmax = -1, 1
        if ymin == ymax == 0:
            ymin, ymax = -1, 1

        assert xmin < xmax
        assert ymin < ymax
        return (xmin, ymin, xmax, ymax)

    def _plot_points(self, bunchs, data_bounds):
        ms = self._marker_size
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
        """Update the view with the selected clusters."""
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return

        # Retrieve the data.
        bunchs = self.coords(self.cluster_ids)
        if bunchs is None:
            self.visual.hide()
            self.canvas.update()
            return
        self.visual.show()
        self.data_bounds = self._get_data_bounds(bunchs)

        self._plot_points(bunchs, self.data_bounds)
        self.canvas.axes.reset_data_bounds(self.data_bounds)
        self.canvas.update()

    def attach(self, gui):
        super(ScatterView, self).attach(gui)
        connect(self.on_request_split)
        self.actions.separator()
        self.actions.add(self.increase)
        self.actions.add(self.decrease)

    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if (self.canvas.lasso.count < 3 or not len(self.cluster_ids)):  # pragma: no cover
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
        pos = range_transform(self.data_bounds, NDC, pos)
        spike_ids = np.concatenate(spike_ids)

        # Find lassoed spikes.
        ind = self.canvas.lasso.in_polygon(pos)
        self.canvas.lasso.clear()
        return np.unique(spike_ids[ind])

    # Marker size
    # -------------------------------------------------------------------------

    @property
    def marker_size(self):
        """Size of the spike markers, in pixels."""
        return self._marker_size

    @marker_size.setter
    def marker_size(self, val):
        assert val > 0
        self._marker_size = val
        self.visual.set_marker_size(val)
        self.canvas.update()

    def increase(self):
        """Increase the marker size."""
        self.marker_size *= self._marker_size_increment

    def decrease(self):
        """Decrease the marker size."""
        self.marker_size = max(.1, self.marker_size / self._marker_size_increment)
