# -*- coding: utf-8 -*-

"""Histogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot.visuals import HistogramVisual, PlotVisual, TextVisual
from phylib.utils.color import selected_cluster_color
from .base import ManualClusteringView, ScalingMixin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Histogram view
# -----------------------------------------------------------------------------

def _compute_histogram(data, x_max=None, x_min=0, n_bins=None, normalize=True, ignore_zeros=False):
    """Compute the histogram of an array."""
    assert x_min <= x_max
    bins = np.linspace(x_min, x_max, n_bins)
    if ignore_zeros:
        data = data[data != 0]
    histogram, _ = np.histogram(data, bins=bins)
    if not normalize:
        return histogram
    # Normalize by the integral of the histogram.
    hist_sum = histogram.sum() * bins[1]
    return histogram / (hist_sum or 1.)


class HistogramView(ScalingMixin, ManualClusteringView):
    """This view displays a histogram for every selected cluster, along with a possible plot
    and some text. To be overriden.

    Constructor
    -----------

    cluster_stat : function
        Maps `cluster_id` to `Bunch(data (1D array), plot (1D array), text)`.

    """

    _default_position = 'right'
    cluster_ids = ()

    # Number of bins in the histogram.
    n_bins = 100

    # Minimum value on the x axis (determines the range of the histogram)
    # If None, then `data.min()` is used.
    x_min = None

    # Maximum value on the x axis (determines the range of the histogram)
    # If None, then `data.max()` is used.
    x_max = None

    # The snippet to update this view are `hn` to change the number of bins, and `hm` to
    # change the maximum value on the x axis. The character `h` can be customized by child classes.
    alias_char = 'h'

    default_shortcuts = {
        'change_window_size': 'ctrl+wheel'
    }

    def __init__(self, cluster_stat=None):
        super(HistogramView, self).__init__()
        self.state_attrs += ('n_bins', 'x_min', 'x_max')
        self.local_state_attrs += ()
        self.canvas.set_layout(layout='stacked', n_plots=1)
        self.canvas.enable_axes()

        self.cluster_stat = cluster_stat

        self.visual = HistogramVisual()
        self.canvas.add_visual(self.visual)

        self.plot_visual = PlotVisual()
        self.canvas.add_visual(self.plot_visual)

        self.text_visual = TextVisual(color=(1., 1., 1., 1.))
        self.canvas.add_visual(self.text_visual)

    def _plot_cluster(self, bunch):
        assert bunch
        n_bins = self.n_bins
        assert n_bins >= 0

        # Update the visual's data.
        self.visual.add_batch_data(
            hist=bunch.histogram, ylim=bunch.ylim, color=bunch.color, box_index=bunch.index)

        # Plot.
        plot = bunch.get('plot', None)
        if plot is not None:
            x = np.linspace(self.x_min, self.x_max, len(plot))
            self.plot_visual.add_batch_data(
                x=x, y=plot, color=(1, 1, 1, 1), data_bounds=self.data_bounds,
                box_index=bunch.index,
            )

        text = bunch.get('text', 'cluster %d' % bunch.cluster_id)
        # Support multiline text.
        text = text.splitlines()
        n = len(text)
        self.text_visual.add_batch_data(
            text=text, pos=[(-1, .8)] * n, anchor=[(1, -1 - 2 * i) for i in range(n)],
            box_index=bunch.index,
        )

    def get_clusters_data(self, load_all=None):
        bunchs = []
        for i, cluster_id in enumerate(self.cluster_ids):
            bunch = self.cluster_stat(cluster_id)
            if not bunch.data.size:
                continue
            bmin, bmax = bunch.data.min(), bunch.data.max()
            # Update self.x_max if it was not set before.
            self.x_min = self.x_min or bunch.get('x_min', None) or bmin
            self.x_max = self.x_max or bunch.get('x_max', None) or bmax
            self.x_min = min(self.x_min, self.x_max)
            assert self.x_min is not None
            assert self.x_max is not None
            assert self.x_min <= self.x_max

            # Compute the histogram.
            bunch.histogram = _compute_histogram(
                bunch.data, x_min=self.x_min, x_max=self.x_max, n_bins=self.n_bins)
            bunch.ylim = bunch.histogram.max()

            bunch.color = selected_cluster_color(i)
            bunch.index = i
            bunch.cluster_id = cluster_id
            bunchs.append(bunch)
        return bunchs

    def _get_data_bounds(self, bunchs):
        # Get the axes data bounds (the last subplot's extended n_cluster times on the y axis).
        ylim = max(bunch.ylim for bunch in bunchs) if bunchs else 1
        return (self.x_min, 0, self.x_max, ylim * len(self.cluster_ids))

    def plot(self, **kwargs):
        """Update the view with the selected clusters."""
        bunchs = self.get_clusters_data()
        self.data_bounds = self._get_data_bounds(bunchs)

        self.canvas.stacked.n_boxes = len(self.cluster_ids)

        self.visual.reset_batch()
        self.plot_visual.reset_batch()
        self.text_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.plot_visual)
        self.canvas.update_visual(self.text_visual)

        self._update_axes()
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(HistogramView, self).attach(gui)

        self.actions.add(
            self.set_n_bins, alias=self.alias_char + 'n',
            prompt=True, prompt_default=lambda: self.n_bins)
        self.actions.add(
            self.set_x_min, alias=self.alias_char + 'min',
            prompt=True, prompt_default=lambda: self.x_min)
        self.actions.add(
            self.set_x_max, alias=self.alias_char + 'max',
            prompt=True, prompt_default=lambda: self.x_max)
        self.actions.separator()

    # Histogram parameters
    # -------------------------------------------------------------------------

    def _get_scaling_value(self):
        return self.x_max

    def _set_scaling_value(self, value):
        self.set_x_max(value)

    def set_n_bins(self, n_bins):
        """Set the number of bins in the histogram."""
        self.n_bins = n_bins
        logger.debug("Change number of bins to %d for %s.", n_bins, self.__class__.__name__)
        self.plot()

    def set_x_min(self, x_min):
        """Set the minimum value on the x axis for the histogram."""
        x_min = min(x_min, self.x_max)
        if x_min == self.x_max:
            return
        self.x_min = x_min
        logger.debug("Change x min to %s for %s.", x_min, self.__class__.__name__)
        self.plot()

    def set_x_max(self, x_max):
        """Set the maximum value on the x axis for the histogram."""
        x_max = max(x_max, self.x_min)
        if x_max == self.x_min:
            return
        self.x_max = x_max
        logger.debug("Change x max to %s for %s.", x_max, self.__class__.__name__)
        self.plot()
