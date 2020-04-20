# -*- coding: utf-8 -*-

"""Correlogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot.transform import Scale
from phy.plot.visuals import HistogramVisual, LineVisual, TextVisual
from phylib.io.array import _clip
from phylib.utils import Bunch
from phy.utils.color import selected_cluster_color, _override_hsv, add_alpha
from .base import ManualClusteringView, ScalingMixin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class CorrelogramView(ScalingMixin, ManualClusteringView):
    """A view showing the autocorrelogram of the selected clusters, and all cross-correlograms
    of cluster pairs.

    Constructor
    -----------

    correlograms : function
        Maps `(cluster_ids, bin_size, window_size)` to an `(n_clusters, n_clusters, n_bins) array`.

    firing_rate : function
        Maps `(cluster_ids, bin_size)` to an `(n_clusters, n_clusters) array`

    """

    # Do not show too many clusters.
    max_n_clusters = 20

    _default_position = 'left'
    cluster_ids = ()

    # Bin size, in seconds.
    bin_size = 1e-3

    # Window size, in seconds.
    window_size = 50e-3

    # Refactory period, in seconds
    refractory_period = 2e-3

    # Whether the normalization is uniform across entire rows or not.
    uniform_normalization = False

    default_shortcuts = {
        'change_window_size': 'ctrl+wheel',
        'change_bin_size': 'alt+wheel',
    }

    default_snippets = {
        'set_bin': 'cb',
        'set_window': 'cw',
        'set_refractory_period': 'cr',
    }

    def __init__(self, correlograms=None, firing_rate=None, sample_rate=None, **kwargs):
        super(CorrelogramView, self).__init__(**kwargs)
        self.state_attrs += (
            'bin_size', 'window_size', 'refractory_period', 'uniform_normalization')
        self.local_state_attrs += ()
        self.canvas.set_layout(layout='grid')

        # Outside margin to show labels.
        self.canvas.gpu_transforms.add(Scale(.9))

        assert sample_rate > 0
        self.sample_rate = float(sample_rate)

        # Function clusters => CCGs.
        self.correlograms = correlograms

        # Function clusters => firing rates (same unit as CCG).
        self.firing_rate = firing_rate

        # Set the default bin and window size.
        self._set_bin_window(bin_size=self.bin_size, window_size=self.window_size)

        self.correlogram_visual = HistogramVisual()
        self.canvas.add_visual(self.correlogram_visual)

        self.line_visual = LineVisual()
        self.canvas.add_visual(self.line_visual)

        self.text_visual = TextVisual(color=(1., 1., 1., 1.))
        self.canvas.add_visual(self.text_visual)

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _iter_subplots(self, n_clusters):
        for i in range(n_clusters):
            for j in range(n_clusters):
                yield i, j

    def get_clusters_data(self, load_all=None):
        ccg = self.correlograms(self.cluster_ids, self.bin_size, self.window_size)
        fr = self.firing_rate(self.cluster_ids, self.bin_size) if self.firing_rate else None
        assert ccg.ndim == 3
        n_bins = ccg.shape[2]
        bunchs = []
        m = ccg.max()
        for i, j in self._iter_subplots(len(self.cluster_ids)):
            b = Bunch()
            b.correlogram = ccg[i, j, :]
            if not self.uniform_normalization:
                # Normalization row per row.
                m = ccg[i, j, :].max()
            b.firing_rate = fr[i, j] if fr is not None else None
            b.data_bounds = (0, 0, n_bins, m)
            b.pair_index = i, j
            b.color = selected_cluster_color(i, 1)
            if i != j:
                b.color = add_alpha(_override_hsv(b.color[:3], s=.1, v=1))
            bunchs.append(b)
        return bunchs

    def _plot_pair(self, bunch):
        # Plot the histogram.
        self.correlogram_visual.add_batch_data(
            hist=bunch.correlogram, color=bunch.color,
            ylim=bunch.data_bounds[3], box_index=bunch.pair_index)

        # Plot the firing rate.
        gray = (.25, .25, .25, 1.)
        if bunch.firing_rate is not None:
            # Line.
            pos = np.array([[0, bunch.firing_rate, bunch.data_bounds[2], bunch.firing_rate]])
            self.line_visual.add_batch_data(
                pos=pos, color=gray, data_bounds=bunch.data_bounds, box_index=bunch.pair_index)
            # # Text.
            # self.text_visual.add_batch_data(
            #     pos=[bunch.data_bounds[2], bunch.firing_rate],
            #     text='%.2f' % bunch.firing_rate,
            #     anchor=(-1, 0),
            #     box_index=bunch.pair_index,
            #     data_bounds=bunch.data_bounds,
            # )

        # Refractory period.
        xrp0 = round((self.window_size * .5 - self.refractory_period) / self.bin_size)
        xrp1 = round((self.window_size * .5 + self.refractory_period) / self.bin_size) + 1
        ylim = bunch.data_bounds[3]
        pos = np.array([[xrp0, 0, xrp0, ylim], [xrp1, 0, xrp1, ylim]])
        self.line_visual.add_batch_data(
            pos=pos, color=gray, data_bounds=bunch.data_bounds, box_index=bunch.pair_index)

    def _plot_labels(self):
        n = len(self.cluster_ids)

        # Display the cluster ids in the subplots.
        for k in range(n):
            self.text_visual.add_batch_data(
                pos=[-1, 0],
                text=str(self.cluster_ids[k]),
                anchor=[-1.25, 0],
                data_bounds=None,
                box_index=(k, 0),
            )
            self.text_visual.add_batch_data(
                pos=[0, -1],
                text=str(self.cluster_ids[k]),
                anchor=[0, -1.25],
                data_bounds=None,
                box_index=(n - 1, k),
            )

        # # Display the window size in the bottom right subplot.
        # self.text_visual.add_batch_data(
        #     pos=[1, -1],
        #     anchor=[1.25, 1],
        #     text='%.1f ms' % (1000 * .5 * self.window_size),
        #     box_index=(n - 1, n - 1),
        # )

    def plot(self, **kwargs):
        """Update the view with the current cluster selection."""
        self.canvas.grid.shape = (len(self.cluster_ids), len(self.cluster_ids))

        bunchs = self.get_clusters_data()

        self.correlogram_visual.reset_batch()
        self.line_visual.reset_batch()
        self.text_visual.reset_batch()

        for bunch in bunchs:
            self._plot_pair(bunch)
        self._plot_labels()

        self.canvas.update_visual(self.correlogram_visual)
        self.canvas.update_visual(self.line_visual)
        self.canvas.update_visual(self.text_visual)

        self.canvas.update()

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def toggle_normalization(self, checked):
        """Change the normalization of the correlograms."""
        self.uniform_normalization = checked
        self.plot()

    def toggle_labels(self, checked):
        """Show or hide all labels."""
        if checked:
            self.text_visual.show()
        else:
            self.text_visual.hide()
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(CorrelogramView, self).attach(gui)

        self.actions.add(self.toggle_normalization, shortcut='n', checkable=True)
        self.actions.add(self.toggle_labels, checkable=True, checked=True)
        self.actions.separator()

        self.actions.add(
            self.set_bin, prompt=True, prompt_default=lambda: self.bin_size * 1000)
        self.actions.add(
            self.set_window, prompt=True, prompt_default=lambda: self.window_size * 1000)
        self.actions.add(
            self.set_refractory_period, prompt=True,
            prompt_default=lambda: self.refractory_period * 1000)
        self.actions.separator()

    # -------------------------------------------------------------------------
    # Methods for changing the parameters
    # -------------------------------------------------------------------------

    def _set_bin_window(self, bin_size=None, window_size=None):
        """Set the bin and window sizes (in seconds)."""
        bin_size = bin_size or self.bin_size
        window_size = window_size or self.window_size
        bin_size = _clip(bin_size, 1e-6, 1e3)
        window_size = _clip(window_size, 1e-6, 1e3)
        assert 1e-6 <= bin_size <= 1e3
        assert 1e-6 <= window_size <= 1e3
        assert bin_size < window_size
        self.bin_size = bin_size
        self.window_size = window_size
        self.update_status()

    @property
    def status(self):
        b, w = self.bin_size * 1000, self.window_size * 1000
        return '{:.1f} ms ({:.1f} ms)'.format(w, b)

    def set_refractory_period(self, value):
        """Set the refractory period (in milliseconds)."""
        self.refractory_period = _clip(value, .1, 100) * 1e-3
        self.plot()

    def set_bin(self, bin_size):
        """Set the correlogram bin size (in milliseconds).

        Example: `1`

        """
        self._set_bin_window(bin_size=bin_size * 1e-3)
        self.plot()

    def set_window(self, window_size):
        """Set the correlogram window size (in milliseconds).

        Example: `100`

        """
        self._set_bin_window(window_size=window_size * 1e-3)
        self.plot()

    def increase(self):
        """Increase the window size."""
        self.set_window(1000 * self.window_size * 1.1)

    def decrease(self):
        """Decrease the window size."""
        self.set_window(1000 * self.window_size / 1.1)

    def on_mouse_wheel(self, e):  # pragma: no cover
        """Change the scaling with the wheel."""
        super(CorrelogramView, self).on_mouse_wheel(e)
        if e.modifiers == ('Alt',):
            self._set_bin_window(bin_size=self.bin_size * 1.1 ** e.delta)
            self.plot()
