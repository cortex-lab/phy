# -*- coding: utf-8 -*-

"""Histogram view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.plot.visuals import HistogramVisual, PlotVisual, TextVisual
from phylib.utils._color import colormaps, _categorical_colormap, add_alpha
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class HistogramView(ManualClusteringView):
    _default_position = 'right'
    cluster_ids = ()

    default_shortcuts = {
    }

    def __init__(self, cluster_stat=None):
        super(HistogramView, self).__init__()
        self.state_attrs += ()
        self.local_state_attrs += ()
        self.canvas.set_layout(layout='stacked', n_plots=1)
        self.canvas.enable_axes()

        # function cluster_id => Bunch(histogram (1D array), plot, text)
        self.cluster_stat = cluster_stat

        self.visual = HistogramVisual()
        self.canvas.add_visual(self.visual)

        self.plot_visual = PlotVisual()
        self.canvas.add_visual(self.plot_visual)

        self.text_visual = TextVisual(color=(1., 1., 1., 1.))
        self.canvas.add_visual(self.text_visual)

    def _plot_cluster(self, idx, cluster_id, bunch=None, color=None, ylim=None, n_clusters=None):
        assert bunch
        n_bins = len(bunch.histogram)
        assert n_bins >= 0
        assert ylim > 0
        data_bounds = (0, 0, n_bins, ylim)

        # Histogram.
        self.visual.add_batch_data(
            hist=bunch.histogram, color=color, ylim=ylim, box_index=idx,
        )

        # Plot.
        plot = bunch.get('plot', None)
        if plot is not None:
            self.plot_visual.add_batch_data(
                x=np.linspace(0., n_bins, len(plot)),
                y=plot,
                color=(1, 1, 1, 1),
                data_bounds=data_bounds,
                box_index=idx,
            )

        text = bunch.get('text', None)
        if text is not None:
            text = text.splitlines()
            n = len(text)
            x = [n_bins * .95] * n
            y = [n_bins * (.9 - .1 * i) for i in range(n)]
            self.text_visual.add_batch_data(
                text=text,
                pos=list(zip(x, y)),
                data_bounds=data_bounds,
                box_index=idx,
            )

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        n_clusters = len(cluster_ids)
        if not cluster_ids:
            return

        bunchs = [self.cluster_stat(cluster_id) for cluster_id in cluster_ids]
        ylim = max(np.max(b.histogram) for b in bunchs)
        # NOTE: We assume, for now, that all histogram of all clusters have the same
        # number of bins.
        n_bins = len(bunchs[0].histogram)

        data_bounds = (0, 0, n_bins, n_clusters * ylim)

        colors = _categorical_colormap(colormaps.default, np.arange(n_clusters))
        colors = add_alpha(colors, 1)

        self.visual.reset_batch()
        self.plot_visual.reset_batch()
        self.text_visual.reset_batch()
        for idx, (cluster_id, bunch) in enumerate(zip(cluster_ids, bunchs)):
            color = colors[idx]
            self._plot_cluster(
                idx, cluster_id, bunch=bunch, color=color, ylim=ylim, n_clusters=n_clusters)
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.plot_visual)
        self.canvas.update_visual(self.text_visual)

        self.canvas.stacked.n_boxes = n_clusters
        self.canvas.axes.reset_data_bounds(data_bounds)
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(HistogramView, self).attach(gui)
