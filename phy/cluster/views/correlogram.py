# -*- coding: utf-8 -*-

"""Correlogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot.transform import Scale
from phy.plot.visuals import HistogramVisual, LineVisual, TextVisual
from phylib.utils.color import _spike_colors
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class CorrelogramView(ManualClusteringView):
    """A view showing the autocorrelogram of the selected clusters, and all cross-correlograms
    of cluster pairs.

    Constructor:

    - `correlograms`: a function
      `(cluster_ids, bin_size, window_size) => (n_clusters, n_clusters, n_bins) array`

    - `firing_rate`: a function `(cluster_ids, bin_size) => (n_clusters, n_clusters) array`

    """

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
        'go_left': 'alt+left',
        'go_right': 'alt+right',
    }

    def __init__(self, correlograms=None, firing_rate=None, sample_rate=None):
        super(CorrelogramView, self).__init__()
        self.state_attrs += (
            'bin_size', 'window_size', 'refractory_period', 'uniform_normalization')
        self.local_state_attrs += (
            'bin_size', 'window_size', 'refractory_period',
        )
        self.canvas.set_layout(layout='grid')

        # Outside margin to show labels.
        self.canvas.transforms.add_on_gpu(Scale(.9))

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

        self.label_visual = TextVisual(color=(1., 1., 1., 1.))
        self.canvas.add_visual(self.label_visual)

    def _iter_subplots(self, n_clusters):
        for i in range(n_clusters):
            for j in range(n_clusters):
                yield i, j

    def _plot_correlograms(self, ccg, ylims=None):
        ylims = ylims or {}
        n_clusters = ccg.shape[0]
        colors = _spike_colors(np.arange(n_clusters), alpha=1.)
        self.correlogram_visual.reset_batch()
        for i, j in self._iter_subplots(n_clusters):
            hist = ccg[i, j, :]
            color = colors[i] if i == j else np.ones(4)
            ylim = ylims.get((i, j), None)
            self.correlogram_visual.add_batch_data(
                hist=hist, color=color, ylim=ylim, box_index=(i, j))
        # Call set_data() after creating the batch.
        self.canvas.update_visual(self.correlogram_visual)

    def _plot_firing_rate(self, fr, ylims=None, n_bins=None):
        assert n_bins > 0
        color = (.25, .25, .25, 1.)
        ylims = ylims or {}
        self.line_visual.reset_batch()
        # Refractory period line coordinates.
        xrp0 = round((self.window_size * .5 - self.refractory_period) / self.bin_size)
        xrp1 = round((self.window_size * .5 + self.refractory_period) / self.bin_size) + 1
        for i, j in self._iter_subplots(len(fr)):
            ylim = ylims.get((i, j), None)
            db = (0, 0, n_bins, ylim)
            f = fr[i, j]
            pos = np.array([[0, f, n_bins, f]])
            # Firing rate.
            self.line_visual.add_batch_data(
                pos=pos, color=color, data_bounds=db, box_index=(i, j))
            # Refractory period.
            pos = np.array([[xrp0, 0, xrp0, ylim], [xrp1, 0, xrp1, ylim]])
            self.line_visual.add_batch_data(
                pos=pos, color=color, data_bounds=db, box_index=(i, j))
        self.canvas.update_visual(self.line_visual)

    def _plot_labels(self, cluster_ids):
        n = len(cluster_ids)
        p = -1.0
        a = -.9
        self.label_visual.reset_batch()
        for k in range(n):
            self.label_visual.add_batch_data(
                pos=[p, 0.],
                text=str(cluster_ids[k]),
                anchor=[a, a],
                data_bounds=None,
                box_index=(k, 0),
            )
            self.label_visual.add_batch_data(
                pos=[0., p],
                text=str(cluster_ids[k]),
                anchor=[a, a],
                data_bounds=None,
                box_index=(n - 1, k),
            )
        self.canvas.update_visual(self.label_visual)

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        n_clusters = len(cluster_ids)
        if not cluster_ids:
            return

        ccg = self.correlograms(
            cluster_ids, self.bin_size, self.window_size)

        # CCG normalization.
        if self.uniform_normalization:
            M = ccg.max()
            ylims = {(i, j): M for i, j in self._iter_subplots(n_clusters)}
        else:
            ylims = {(i, j): ccg[i, j, :].max() for i, j in self._iter_subplots(n_clusters)}

        self.canvas.grid.shape = (n_clusters, n_clusters)
        self._plot_correlograms(ccg, ylims=ylims)

        # Show firing rate as horizontal lines.
        if self.firing_rate:
            fr = self.firing_rate(cluster_ids, self.bin_size)
            self._plot_firing_rate(fr, ylims=ylims, n_bins=ccg.shape[2])

        self._plot_labels(cluster_ids)

    def toggle_normalization(self, checked):
        """Change the normalization of the correlograms."""
        self.uniform_normalization = checked
        self.on_select(cluster_ids=self.cluster_ids)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(CorrelogramView, self).attach(gui)
        self.actions.add(self.toggle_normalization, shortcut='n', checkable=True)
        self.actions.separator()
        self.actions.add(
            self.set_bin, alias='cb', prompt=True,
            prompt_default=lambda: self.bin_size * 1000)
        self.actions.add(
            self.set_window, alias='cw', prompt=True,
            prompt_default=lambda: self.window_size * 1000)
        self.actions.add(
            self.set_refractory_period, alias='cr', prompt=True,
            prompt_default=lambda: self.refractory_period * 1000)

    def _set_bin_window(self, bin_size=None, window_size=None):
        """Set the bin and window sizes (in seconds)."""
        bin_size = bin_size or self.bin_size
        window_size = window_size or self.window_size
        assert 1e-6 < bin_size < 1e3
        assert 1e-6 < window_size < 1e3
        assert bin_size < window_size
        self.bin_size = bin_size
        self.window_size = window_size
        # Set the status message.
        b, w = self.bin_size * 1000, self.window_size * 1000
        self.set_status('Bin: {:.1f} ms. Window: {:.1f} ms.'.format(b, w))

    def set_refractory_period(self, value):
        """Set the refractory period (in milliseconds)."""
        self.refractory_period = np.clip(value, .1, 100) * 1e-3
        self.on_select(cluster_ids=self.cluster_ids)

    def set_bin(self, bin_size):
        """Set the correlogram bin size (in milliseconds).

        Example: `1`

        """
        self._set_bin_window(bin_size=bin_size * 1e-3)
        self.on_select(cluster_ids=self.cluster_ids)

    def set_window(self, window_size):
        """Set the correlogram window size (in milliseconds).

        Example: `100`

        """
        self._set_bin_window(window_size=window_size * 1e-3)
        self.on_select(cluster_ids=self.cluster_ids)

    def increase(self):
        self.set_window(1000 * self.window_size * 1.1)

    def decrease(self):
        self.set_window(1000 * self.window_size / 1.1)
