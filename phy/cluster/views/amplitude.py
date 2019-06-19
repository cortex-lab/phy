# -*- coding: utf-8 -*-

"""Amplitude view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils.color import selected_cluster_color
from phylib.utils._types import _as_array
from .base import ManualClusteringView, MarkerSizeMixin, LassoMixin
from phy.plot.visuals import ScatterVisual, TextVisual

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Amplitude view
# -----------------------------------------------------------------------------

class AmplitudeView(MarkerSizeMixin, LassoMixin, ManualClusteringView):
    """This view displays an amplitude plot for all selected clusters.

    Constructor
    -----------

    amplitudes : function
        Maps `cluster_ids` to a list `[Bunch(amp, spike_ids), ...]` for each cluster.

    """

    _default_position = 'right'
    default_shortcuts = {
        'next_amplitude_type': 'a',
    }

    def __init__(self, amplitudes=None, amplitude_name=None, spike_times=None, duration=None):
        super(AmplitudeView, self).__init__()
        # Save the marker size in the global and local view's config.

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

        self.spike_times = _as_array(spike_times)
        assert spike_times is not None
        self.duration = duration or (spike_times[-1] if len(spike_times) else 1.)

        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

    def _get_data_bounds(self, bunchs):
        """Compute the data bounds."""
        M = max(bunch.amp.max() for bunch in bunchs) if bunchs else 1.
        return (0, 0, self.duration, M)

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids."""
        bunchs = self.amplitudes[self.amplitude_name](self.cluster_ids, load_all=load_all) or ()
        # Add a pos attribute in bunchs in addition to x and y.
        for i, bunch in enumerate(bunchs):
            spike_ids = _as_array(bunch.spike_ids)
            spike_times = self.spike_times[spike_ids]
            amplitudes = _as_array(bunch.amp)
            assert spike_ids.shape == spike_times.shape == amplitudes.shape
            bunch.pos = np.c_[spike_times, amplitudes]
            assert bunch.pos.ndim == 2
            bunch.color = selected_cluster_color(i, .75)
        return bunchs

    def _plot_cluster(self, bunch):
        """Make the scatter plot."""
        ms = self._marker_size
        # Scatter plot.
        self.visual.add_batch_data(
            pos=bunch.pos, color=bunch.color, size=ms, data_bounds=self.data_bounds)

    def _plot_amplitude_name(self):
        """Show the amplitude name."""
        self.text_visual.add_batch_data(pos=[1, 1], anchor=[-1, -1], text=self.amplitude_name)

    def plot(self):
        """Update the view with the current cluster selection."""
        bunchs = self.get_clusters_data()
        self.data_bounds = self._get_data_bounds(bunchs)

        self.visual.reset_batch()
        self.text_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self._plot_amplitude_name()
        self.canvas.update_visual(self.visual)
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
