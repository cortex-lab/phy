# -*- coding: utf-8 -*-

"""Histogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot.visuals import HistogramVisual, PlotVisual, TextVisual
from phylib.utils.color import colormaps, _categorical_colormap, add_alpha
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class HistogramView(ManualClusteringView):
    """This view displays a histogram for every selected cluster.

    Constructor:

    - `cluster_stat`: a function `cluster_id => Bunch(histogram (1D array), plot (1D array), text)`

    """

    _default_position = 'right'
    cluster_ids = ()

    # Number of bins in the histogram.
    n_bins = 100

    # Maximum value on the x axis (determines the range of the histogram)
    # If None, then `data.max()` is used.
    x_max = None

    # The snippet to update this view are `hn` to change the number of bins, and `hm` to
    # change the maximum value on the x axis. The character `h` can be customized by child classes.
    alias_char = 'h'

    default_shortcuts = {
    }

    def __init__(self, cluster_stat=None):
        super(HistogramView, self).__init__()
        self.state_attrs += ('n_bins', 'x_max')
        self.local_state_attrs += ('n_bins', 'x_max')
        self.canvas.set_layout(layout='stacked', n_plots=1)
        self.canvas.enable_axes()

        self.cluster_stat = cluster_stat
        self._hist_max = None

        self.visual = HistogramVisual()
        self.canvas.add_visual(self.visual)

        self.plot_visual = PlotVisual()
        self.canvas.add_visual(self.plot_visual)

        self.text_visual = TextVisual(color=(1., 1., 1., 1.))
        self.canvas.add_visual(self.text_visual)

    def _plot_cluster(
            self, idx, cluster_id, bunch=None, color=None, n_clusters=None):
        assert bunch
        n_bins = self.n_bins
        assert n_bins >= 0

        # Update self.x_max if it was not set before.
        self.x_max = self.x_max or bunch.get('x_max', None) or bunch.data.max()
        assert self.x_max is not None
        assert self.x_max > 0

        # Compute the histogram.
        bins = np.linspace(0., self.x_max, self.n_bins)
        histogram, _ = np.histogram(bunch.data, bins=bins)

        # Normalize by the integral of the histogram.
        hist_sum = histogram.sum() * bins[1]
        histogram = histogram / (hist_sum or 1.)
        self._hist_max = histogram.max()
        data_bounds = (0, 0, self.x_max, self._hist_max)

        # Update the visual's data.
        self.visual.add_batch_data(
            hist=histogram, ylim=self._hist_max, color=color, box_index=idx)

        # Plot.
        plot = bunch.get('plot', None)
        if plot is not None:
            x = np.linspace(0., self.x_max, len(plot))
            self.plot_visual.add_batch_data(
                x=x,
                y=plot,
                color=(1, 1, 1, 1),
                data_bounds=data_bounds,
                box_index=idx,
            )

        text = bunch.get('text', 'cluster %d' % cluster_id)
        # Support multiline text.
        text = text.splitlines()
        n = len(text)
        x = [-.75] * n
        y = [+.8 - i * .25 for i in range(n)]  # improve positioning of text
        self.text_visual.add_batch_data(
            text=text,
            pos=list(zip(x, y)),
            box_index=idx,
        )

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        n_clusters = len(cluster_ids)
        if not cluster_ids:
            return

        bunchs = [self.cluster_stat(cluster_id) for cluster_id in cluster_ids]

        # Cluster colors.
        colors = _categorical_colormap(colormaps.default, np.arange(n_clusters))
        colors = add_alpha(colors, 1)

        self.canvas.stacked.n_boxes = n_clusters

        self.visual.reset_batch()
        self.plot_visual.reset_batch()
        self.text_visual.reset_batch()
        for idx, (cluster_id, bunch) in enumerate(zip(cluster_ids, bunchs)):
            color = colors[idx]
            self._plot_cluster(
                idx, cluster_id, bunch=bunch, color=color, n_clusters=n_clusters)
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.plot_visual)
        self.canvas.update_visual(self.text_visual)

        # Get the axes data bounds (the last subplot's extended n_cluster times on the y axis).
        data_bounds = (0, 0, self.x_max, self._hist_max * n_clusters)
        self.canvas.axes.reset_data_bounds(data_bounds)
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(HistogramView, self).attach(gui)
        self.actions.add(
            self.set_n_bins, alias=self.alias_char + 'n',
            prompt=True, prompt_default=lambda: self.n_bins)
        self.actions.add(
            self.set_x_max, alias=self.alias_char + 'm',
            prompt=True, prompt_default=lambda: self.x_max)

    def set_n_bins(self, n_bins):
        """Set the number of bins in the histogram."""
        self.n_bins = n_bins
        logger.debug("Change number of bins to %d for %s.", n_bins, self.__class__.__name__)
        self.on_select(cluster_ids=self.cluster_ids)

    def set_x_max(self, x_max):
        """Set the maximum value on the x axis for the histogram."""
        self.x_max = x_max
        logger.debug("Change x max to %s for %s.", x_max, self.__class__.__name__)
        self.on_select(cluster_ids=self.cluster_ids)

    def increase(self):
        self.set_x_max(self.x_max * 1.1)

    def decrease(self):
        self.set_x_max(self.x_max / 1.1)
