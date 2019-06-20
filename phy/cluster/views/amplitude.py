# -*- coding: utf-8 -*-

"""Amplitude view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils.color import selected_cluster_color, add_alpha
from phylib.utils._types import _as_array

from .base import ManualClusteringView, MarkerSizeMixin, LassoMixin
from .histogram import _compute_histogram
from phy.plot.transform import Rotate, Range, NDC
from phy.plot.visuals import ScatterVisual, TextVisual, HistogramVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Amplitude view
# -----------------------------------------------------------------------------

class AmplitudeView(MarkerSizeMixin, LassoMixin, ManualClusteringView):
    """This view displays an amplitude plot for all selected clusters.

    Constructor
    -----------

    amplitudes : function
        Maps `cluster_ids` to a list `[Bunch(amplitudes, spike_ids), ...]` for each cluster.
        Use `cluster_id=None` for background amplitudes.

    """

    _default_position = 'right'

    # Alpha channel of the markers in the scatter plot.
    marker_alpha = 1.

    # Number of bins in the histogram.
    n_bins = 100

    # Alpha channel of the histogram in the background.
    histogram_alpha = .5

    # Size of the histogram, between 0 and 1.
    histogram_scale = .25

    default_shortcuts = {
        'next_amplitude_type': 'a',
    }

    def __init__(self, amplitudes=None, amplitude_name=None, duration=None):
        super(AmplitudeView, self).__init__()
        self.state_attrs += ('amplitude_name',)

        self.canvas.enable_axes()
        self.canvas.enable_lasso()
        # Ensure amplitudes is a dictionary, even if there is a single amplitude.
        if not isinstance(amplitudes, dict):
            amplitudes = {'amplitude': amplitudes}
        assert amplitudes
        self.amplitudes = amplitudes
        self.amplitude_names = list(amplitudes.keys())
        # Current amplitude type.
        self.amplitude_name = amplitude_name or self.amplitude_names[0]
        assert self.amplitude_name in amplitudes

        self.cluster_ids = ()
        self.duration = duration or 1

        # Histogram visual.
        self.hist_visual = HistogramVisual()
        self.hist_visual.transforms.add_on_gpu([
            Range(NDC, (-1, -1, 1, -1 + 2 * self.histogram_scale)), Rotate('ccw')])
        self.canvas.add_visual(self.hist_visual)

        # Scatter plot.
        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

        # Amplitude name.
        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

    def _get_data_bounds(self, bunchs):
        """Compute the data bounds."""
        m = min(np.min(bunch.amplitudes) for bunch in bunchs) if bunchs else 0.
        m = min(0, m)  # ensure ymin <= 0
        M = max(np.max(bunch.amplitudes) for bunch in bunchs) if bunchs else 1.
        return (0, m, self.duration, M)

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids."""
        if not len(self.cluster_ids):
            return
        cluster_ids = list(self.cluster_ids)
        # Don't need the background when splitting.
        if not load_all:
            # Add None cluster which means background spikes.
            cluster_ids = [None] + cluster_ids
        bunchs = self.amplitudes[self.amplitude_name](cluster_ids, load_all=load_all) or ()
        # Add a pos attribute in bunchs in addition to x and y.
        for i, (cluster_id, bunch) in enumerate(zip(cluster_ids, bunchs)):
            spike_ids = _as_array(bunch.spike_ids)
            spike_times = _as_array(bunch.spike_times)
            amplitudes = _as_array(bunch.amplitudes)
            assert spike_ids.shape == spike_times.shape == amplitudes.shape
            # Ensure that bunch.pos exists, as it used by the LassoMixin.
            bunch.pos = np.c_[spike_times, amplitudes]
            assert bunch.pos.ndim == 2
            bunch.cluster_id = cluster_id
            bunch.color = (
                selected_cluster_color(i - 1, self.marker_alpha)
                # Background amplitude color.
                if cluster_id is not None else (.5, .5, .5, .5))
        return bunchs

    def _add_histograms(self, bunchs):
        # We do this after get_clusters_data because we need x_max.
        for bunch in bunchs:
            bunch.histogram = _compute_histogram(
                bunch.amplitudes,
                x_min=self.data_bounds[1],
                x_max=self.data_bounds[3],
                n_bins=self.n_bins,
                normalize=False)
        return bunchs

    def _plot_cluster(self, bunch):
        """Make the scatter plot."""
        ms = self._marker_size

        # Histogram in the background.
        self.hist_visual.add_batch_data(
            hist=bunch.histogram,
            ylim=self._ylim,
            color=add_alpha(bunch.color, self.histogram_alpha))

        # Scatter plot.
        self.visual.add_batch_data(
            pos=bunch.pos, color=bunch.color, size=ms, data_bounds=self.data_bounds)

    def _plot_amplitude_name(self):
        """Show the amplitude name."""
        self.text_visual.add_batch_data(pos=[1, 1], anchor=[-1, -1], text=self.amplitude_name)

    def plot(self, **kwargs):
        """Update the view with the current cluster selection."""
        bunchs = self.get_clusters_data()
        if not bunchs:
            return
        self.data_bounds = self._get_data_bounds(bunchs)
        bunchs = self._add_histograms(bunchs)
        # Use the same scale for all histograms.
        self._ylim = max(bunch.histogram.max() for bunch in bunchs) if bunchs else 1.

        self.visual.reset_batch()
        self.hist_visual.reset_batch()
        self.text_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self._plot_amplitude_name()
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.hist_visual)
        self.canvas.update_visual(self.text_visual)

        self._update_axes()
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(AmplitudeView, self).attach(gui)
        self.actions.add(self.next_amplitude_type)

    def next_amplitude_type(self):
        """Switch to the next amplitude type."""
        i = self.amplitude_names.index(self.amplitude_name)
        n = len(self.amplitude_names)
        self.amplitude_name = self.amplitude_names[(i + 1) % n]
        logger.debug("Switch to amplitude type %s.", self.amplitude_name)
        self.plot()
