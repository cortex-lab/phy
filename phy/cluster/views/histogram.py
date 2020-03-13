# -*- coding: utf-8 -*-

"""Histogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.io.array import _clip
from phy.plot.visuals import HistogramVisual, TextVisual
from phy.utils.color import selected_cluster_color
from .base import ManualClusteringView, ScalingMixin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Histogram view
# -----------------------------------------------------------------------------

def _compute_histogram(
        data, x_max=None, x_min=None, n_bins=None, normalize=True, ignore_zeros=False):
    """Compute the histogram of an array."""
    assert x_min <= x_max
    assert n_bins >= 0
    n_bins = _clip(n_bins, 2, 1000000)
    bins = np.linspace(float(x_min), float(x_max), int(n_bins))
    if ignore_zeros:
        data = data[data != 0]
    histogram, _ = np.histogram(data, bins=bins)
    if not normalize:  # pragma: no cover
        return histogram
    # Normalize by the integral of the histogram.
    hist_sum = histogram.sum() * (bins[1] - bins[0])
    return histogram / (hist_sum or 1.)


def _first_not_null(*l):
    for x in l:
        if x is not None:
            return x


class HistogramView(ScalingMixin, ManualClusteringView):
    """This view displays a histogram for every selected cluster, along with a possible plot
    and some text. To be overriden.

    Constructor
    -----------

    cluster_stat : function
        Maps `cluster_id` to `Bunch(data (1D array), plot (1D array), text)`.

    """

    # Do not show too many clusters.
    max_n_clusters = 20

    _default_position = 'right'
    cluster_ids = ()

    # Number of bins in the histogram.
    n_bins = 100

    # Step on the x axis when changing the histogram range with the mouse wheel.
    x_delta = .01  # in seconds

    # Minimum value on the x axis (determines the range of the histogram)
    # If None, then `data.min()` is used.
    x_min = None

    # Maximum value on the x axis (determines the range of the histogram)
    # If None, then `data.max()` is used.
    x_max = None

    # Unit of the bin in the set_bin_size, set_x_min, set_x_max actions.
    bin_unit = 's'  # s (seconds) or ms (milliseconds)

    # The snippet to update this view are `hn` to change the number of bins, and `hm` to
    # change the maximum value on the x axis. The character `h` can be customized by child classes.
    alias_char = 'h'

    default_shortcuts = {
        'change_window_size': 'ctrl+wheel',
    }

    default_snippets = {
        'set_n_bins': '%sn' % alias_char,
        'set_bin_size (%s)' % bin_unit: '%sb' % alias_char,
        'set_x_min (%s)' % bin_unit: '%smin' % alias_char,
        'set_x_max (%s)' % bin_unit: '%smax' % alias_char,
    }

    _state_attrs = ('n_bins', 'x_min', 'x_max')
    _local_state_attrs = ()

    def __init__(self, cluster_stat=None):
        super(HistogramView, self).__init__()
        self.state_attrs += self._state_attrs
        self.local_state_attrs += self._local_state_attrs
        self.canvas.set_layout(layout='stacked', n_plots=1)
        self.canvas.enable_axes()

        self.cluster_stat = cluster_stat

        self.visual = HistogramVisual()
        self.canvas.add_visual(self.visual)

        # self.plot_visual = PlotVisual()
        # self.canvas.add_visual(self.plot_visual)

        self.text_visual = TextVisual(color=(1., 1., 1., 1.))
        self.canvas.add_visual(self.text_visual)

    def _plot_cluster(self, bunch):
        assert bunch
        n_bins = self.n_bins
        assert n_bins >= 0

        # Update the visual's data.
        self.visual.add_batch_data(
            hist=bunch.histogram, ylim=bunch.ylim, color=bunch.color, box_index=bunch.index)

        # # Plot.
        # plot = bunch.get('plot', None)
        # if plot is not None:
        #     x = np.linspace(self.x_min, self.x_max, len(plot))
        #     self.plot_visual.add_batch_data(
        #         x=x, y=plot, color=(1, 1, 1, 1), data_bounds=self.data_bounds,
        #         box_index=bunch.index,
        #     )

        text = bunch.get('text', None)
        if not text:
            return
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
            self.x_min = _first_not_null(self.x_min, bunch.get('x_min', None), bmin)
            self.x_max = _first_not_null(self.x_max, bunch.get('x_max', None), bmax)
            self.x_min, self.x_max = sorted((self.x_min, self.x_max))
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
        # self.plot_visual.reset_batch()
        self.text_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.visual)
        # self.canvas.update_visual(self.plot_visual)
        self.canvas.update_visual(self.text_visual)

        self._update_axes()
        self.canvas.update()
        self.update_status()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(HistogramView, self).attach(gui)

        self.actions.add(
            self.set_n_bins, alias=self.alias_char + 'n',
            prompt=True, prompt_default=lambda: self.n_bins)
        self.actions.add(
            self.set_bin_size, alias=self.alias_char + 'b',
            prompt=True, prompt_default=lambda: self.bin_size)
        self.actions.add(
            self.set_x_min, alias=self.alias_char + 'min',
            prompt=True, prompt_default=lambda: self.x_min)
        self.actions.add(
            self.set_x_max, alias=self.alias_char + 'max',
            prompt=True, prompt_default=lambda: self.x_max)
        self.actions.separator()

    @property
    def status(self):
        f = 1 if self.bin_unit == 's' else 1000
        return '[{:.1f}{u}, {:.1f}{u:s}]'.format(
            (self.x_min or 0) * f, (self.x_max or 0) * f, u=self.bin_unit)

    # Histogram parameters
    # -------------------------------------------------------------------------

    def _get_scaling_value(self):
        return self.x_max

    def _set_scaling_value(self, value):
        if self.bin_unit == 'ms':
            value *= 1000
        self.set_x_max(value)

    def set_n_bins(self, n_bins):
        """Set the number of bins in the histogram."""
        self.n_bins = n_bins
        logger.debug("Change number of bins to %d for %s.", n_bins, self.__class__.__name__)
        self.plot()

    @property
    def bin_size(self):
        """Return the bin size (in seconds or milliseconds depending on `self.bin_unit`)."""
        bs = (self.x_max - self.x_min) / self.n_bins
        if self.bin_unit == 'ms':
            bs *= 1000
        return bs

    def set_bin_size(self, bin_size):
        """Set the bin size in the histogram."""
        assert bin_size > 0
        if self.bin_unit == 'ms':
            bin_size /= 1000
        self.n_bins = np.round((self.x_max - self.x_min) / bin_size)
        logger.debug("Change number of bins to %d for %s.", self.n_bins, self.__class__.__name__)
        self.plot()

    def set_x_min(self, x_min):
        """Set the minimum value on the x axis for the histogram."""
        if self.bin_unit == 'ms':
            x_min /= 1000
        x_min = min(x_min, self.x_max)
        if x_min == self.x_max:
            return
        self.x_min = x_min
        logger.log(5, "Change x min to %s for %s.", x_min, self.__class__.__name__)
        self.plot()

    def set_x_max(self, x_max):
        """Set the maximum value on the x axis for the histogram."""
        if self.bin_unit == 'ms':
            x_max /= 1000
        x_max = max(x_max, self.x_min)
        if x_max == self.x_min:
            return
        self.x_max = x_max
        logger.log(5, "Change x max to %s for %s.", x_max, self.__class__.__name__)
        self.plot()

    def on_mouse_wheel(self, e):  # pragma: no cover
        """Change the scaling with the wheel."""
        super(HistogramView, self).on_mouse_wheel(e)
        if e.modifiers == ('Shift',):
            self.x_min *= 1.1 ** e.delta
            self.x_min = min(self.x_min, self.x_max)
            if self.x_min < self.x_max:
                self.plot()
        elif e.modifiers == ('Alt',):
            self.n_bins /= 1.05 ** e.delta
            self.n_bins = int(self.n_bins)
            self.n_bins = max(2, self.n_bins)
            self.plot()


class ISIView(HistogramView):
    """Histogram view showing the interspike intervals."""
    x_min = 0
    x_max = .05  # window size is 50 ms by default
    n_bins = int(x_max / .001)  # by default, 1 bin = 1 ms
    alias_char = 'isi'  # provide `:isisn` (set number of bins) and `:isim` (set max bin) snippets
    bin_unit = 'ms'  # user-provided bin values in milliseconds, but stored in seconds

    default_shortcuts = {
        'change_window_size': 'ctrl+wheel',
    }

    default_snippets = {
        'set_n_bins': '%sn' % alias_char,
        'set_bin_size (%s)' % bin_unit: '%sb' % alias_char,
        'set_x_min (%s)' % bin_unit: '%smin' % alias_char,
        'set_x_max (%s)' % bin_unit: '%smax' % alias_char,
    }


class FiringRateView(HistogramView):
    """Histogram view showing the time-dependent firing rate."""
    n_bins = 200
    alias_char = 'fr'
    bin_unit = 's'
    x_min = 0

    _state_attrs = ('n_bins', 'x_min')
    _local_state_attrs = ('x_max',)  # depends on the duration of the dataset

    default_shortcuts = {
        'change_window_size': 'ctrl+wheel',
    }

    default_snippets = {
        'set_n_bins': '%sn' % alias_char,
        'set_bin_size (%s)' % bin_unit: '%sb' % alias_char,
        'set_x_min (%s)' % bin_unit: '%smin' % alias_char,
        'set_x_max (%s)' % bin_unit: '%smax' % alias_char,
    }
