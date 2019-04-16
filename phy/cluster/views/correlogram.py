# -*- coding: utf-8 -*-

"""Correlogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot.transform import Scale
from phy.utils._color import _spike_colors
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class CorrelogramView(ManualClusteringView):
    _callback_delay = 30
    _default_position = 'left'
    cluster_ids = ()

    bin_size = 1e-3
    window_size = 50e-3
    uniform_normalization = False

    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
    }

    def __init__(self, correlograms=None, sample_rate=None):
        super(CorrelogramView, self).__init__()
        self.canvas.set_layout(layout='grid')
        # Outside margin to show labels.
        self.canvas.transforms.add_on_gpu(Scale(.9))

        assert sample_rate > 0
        self.sample_rate = float(sample_rate)

        # Function clusters => CCGs.
        self.correlograms = correlograms

        # Set the default bin and window size.
        self.set_bin_window(bin_size=self.bin_size,
                            window_size=self.window_size)

    def set_bin_window(self, bin_size=None, window_size=None):
        """Set the bin and window sizes."""
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

    def _iter_subplots(self, n_clusters):
        for i in range(n_clusters):
            for j in range(n_clusters):
                yield i, j

    def _plot_correlograms(self, ccg):
        n_clusters = ccg.shape[0]
        ylim = [ccg.max()] if not self.uniform_normalization else None
        colors = _spike_colors(np.arange(n_clusters), alpha=1.)
        for i, j in self._iter_subplots(n_clusters):
            hist = ccg[i, j, :]
            color = colors[i] if i == j else np.ones(4)
            self.canvas[i, j].hist(hist, color=color, ylim=ylim)

    def _plot_labels(self, cluster_ids):
        n = len(cluster_ids)
        for k in range(n):
            self.canvas[k, 0].text_batch(
                pos=[-1., 0.],
                text=str(cluster_ids[k]),
                anchor=[-0.90, -0.90],
                data_bounds=None,
            )
            self.canvas[n - 1, k].text_batch(
                pos=[0., -1.],
                text=str(cluster_ids[k]),
                anchor=[-0.90, -0.90],
                data_bounds=None,
            )
        self.canvas.text(color=(1., 1., 1., 1.))

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        n_clusters = len(cluster_ids)
        if not cluster_ids:
            return

        ccg = self.correlograms(
            cluster_ids, self.bin_size, self.window_size)

        self.canvas.grid.shape = (n_clusters, n_clusters)
        self.canvas.clear()
        self._plot_correlograms(ccg)
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
        self.actions.add(self.set_bin, alias='cb')
        self.actions.add(self.set_window, alias='cw')

    @property
    def state(self):
        state = super(CorrelogramView, self).state
        state.update(bin_size=self.bin_size,
                     window_size=self.window_size,
                     uniform_normalization=self.uniform_normalization,
                     )
        return state

    def set_bin(self, bin_size):
        """Set the correlogram bin size (in milliseconds).

        Example: `1`

        """
        self.set_bin_window(bin_size=bin_size * 1e-3)
        self.on_select(cluster_ids=self.cluster_ids)

    def set_window(self, window_size):
        """Set the correlogram window size (in milliseconds).

        Example: `100`

        """
        self.set_bin_window(window_size=window_size * 1e-3)
        self.on_select(cluster_ids=self.cluster_ids)
