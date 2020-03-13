# -*- coding: utf-8 -*-

"""Amplitude view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils.color import selected_cluster_color, add_alpha
from phylib.utils._types import _as_array
from phylib.utils.event import emit

from phy.cluster._utils import RotatingProperty
from phy.plot.transform import Rotate, Scale, Translate, Range, NDC
from phy.plot.visuals import ScatterVisual, HistogramVisual, PatchVisual
from .base import ManualClusteringView, MarkerSizeMixin, LassoMixin
from .histogram import _compute_histogram

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Amplitude view
# -----------------------------------------------------------------------------

class AmplitudeView(MarkerSizeMixin, LassoMixin, ManualClusteringView):
    """This view displays an amplitude plot for all selected clusters.

    Constructor
    -----------

    amplitudes : dict
        Dictionary `{amplitudes_type: function}`, for different types of amplitudes.

        Each function maps `cluster_ids` to a list
        `[Bunch(amplitudes, spike_ids, spike_times), ...]` for each cluster.
        Use `cluster_id=None` for background amplitudes.

    """

    # Do not show too many clusters.
    max_n_clusters = 8

    _default_position = 'right'

    # Alpha channel of the markers in the scatter plot.
    marker_alpha = 1.
    time_range_color = (1., 1., 0., .25)

    # Number of bins in the histogram.
    n_bins = 100

    # Alpha channel of the histogram in the background.
    histogram_alpha = .5

    # Quantile used for scaling of the amplitudes (less than 1 to avoid outliers).
    quantile = .99

    # Size of the histogram, between 0 and 1.
    histogram_scale = .25

    default_shortcuts = {
        'change_marker_size': 'alt+wheel',
        'next_amplitudes_type': 'a',
        'previous_amplitudes_type': 'shift+a',
        'select_x_dim': 'shift+left click',
        'select_y_dim': 'shift+right click',
        'select_time': 'alt+click',
    }

    def __init__(self, amplitudes=None, amplitudes_type=None, duration=None):
        super(AmplitudeView, self).__init__()
        self.state_attrs += ('amplitudes_type',)

        self.canvas.enable_axes()
        self.canvas.enable_lasso()

        # Ensure amplitudes is a dictionary, even if there is a single amplitude.
        if not isinstance(amplitudes, dict):
            amplitudes = {'amplitude': amplitudes}
        assert amplitudes
        self.amplitudes = amplitudes

        # Rotating property amplitudes types.
        self.amplitudes_types = RotatingProperty()
        for name, value in self.amplitudes.items():
            self.amplitudes_types.add(name, value)
        # Current amplitudes type.
        self.amplitudes_types.set(amplitudes_type)
        assert self.amplitudes_type in self.amplitudes

        self.cluster_ids = ()
        self.duration = duration or 1.

        # Histogram visual.
        self.hist_visual = HistogramVisual()
        self.hist_visual.transforms.add([
            Range(NDC, (-1, -1, 1, -1 + 2 * self.histogram_scale)),
            Rotate('cw'),
            Scale((1, -1)),
            Translate((2.05, 0)),
        ])
        self.canvas.add_visual(self.hist_visual)
        self.canvas.panzoom.zoom = self.canvas.panzoom._default_zoom = (.75, 1)
        self.canvas.panzoom.pan = self.canvas.panzoom._default_pan = (-.25, 0)

        # Yellow vertical bar showing the selected time interval.
        self.patch_visual = PatchVisual(primitive_type='triangle_fan')
        self.patch_visual.inserter.insert_vert('''
            const float MIN_INTERVAL_SIZE = 0.01;
            uniform float u_interval_size;
        ''', 'header')
        self.patch_visual.inserter.insert_vert('''
            gl_Position.y = pos_orig.y;

            // The following is used to ensure that (1) the bar width increases with the zoom level
            // but also (2) there is a minimum absolute width so that the bar remains visible
            // at low zoom levels.
            float w = max(MIN_INTERVAL_SIZE, u_interval_size * u_zoom.x);
            // HACK: the z coordinate is used to store 0 or 1, depending on whether the current
            // vertex is on the left or right edge of the bar.
            gl_Position.x += w * (-1 + 2 * int(a_position.z == 0));

        ''', 'after_transforms')
        self.canvas.add_visual(self.patch_visual)

        # Scatter plot.
        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)
        self.canvas.panzoom.set_constrain_bounds((-2, -2, +2, +2))

    def _get_data_bounds(self, bunchs):
        """Compute the data bounds."""
        if not bunchs:  # pragma: no cover
            return (0, 0, self.duration, 1)
        m = min(
            np.quantile(bunch.amplitudes, 1 - self.quantile)
            for bunch in bunchs if len(bunch.amplitudes))
        m = min(0, m)  # ensure ymin <= 0
        M = max(
            np.quantile(bunch.amplitudes, self.quantile)
            for bunch in bunchs if len(bunch.amplitudes))
        return (0, m, self.duration, M)

    def _add_histograms(self, bunchs):
        # We do this after get_clusters_data because we need x_max.
        for bunch in bunchs:
            bunch.histogram = _compute_histogram(
                bunch.amplitudes,
                x_min=self.data_bounds[1],
                x_max=self.data_bounds[3],
                n_bins=self.n_bins,
                normalize=True,
                ignore_zeros=True,
            )
        return bunchs

    def show_time_range(self, interval=(0, 0)):
        start, end = interval
        x0 = -1 + 2 * (start / self.duration)
        x1 = -1 + 2 * (end / self.duration)
        xm = .5 * (x0 + x1)
        pos = np.array([
            [xm, -1],
            [xm, +1],
            [xm, +1],
            [xm, -1],
        ])
        self.patch_visual.program['u_interval_size'] = .5 * (x1 - x0)
        self.patch_visual.set_data(pos=pos, color=self.time_range_color, depth=[0, 0, 1, 1])
        self.canvas.update()

    def _plot_cluster(self, bunch):
        """Make the scatter plot."""
        ms = self._marker_size
        if not len(bunch.histogram):
            return

        # Histogram in the background.
        self.hist_visual.add_batch_data(
            hist=bunch.histogram,
            ylim=self._ylim,
            color=add_alpha(bunch.color, self.histogram_alpha))

        # Scatter plot.
        self.visual.add_batch_data(
            pos=bunch.pos, color=bunch.color, size=ms, data_bounds=self.data_bounds)

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids."""
        if not len(self.cluster_ids):
            return
        cluster_ids = list(self.cluster_ids)
        # Don't need the background when splitting.
        if not load_all:
            # Add None cluster which means background spikes.
            cluster_ids = [None] + cluster_ids
        bunchs = self.amplitudes[self.amplitudes_type](cluster_ids, load_all=load_all) or ()
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

    def plot(self, **kwargs):
        """Update the view with the current cluster selection."""
        bunchs = self.get_clusters_data(**kwargs)
        if not bunchs:
            return
        self.data_bounds = self._get_data_bounds(bunchs)
        bunchs = self._add_histograms(bunchs)
        # Use the same scale for all histograms.
        self._ylim = max(bunch.histogram.max() for bunch in bunchs) if bunchs else 1.

        self.visual.reset_batch()
        self.hist_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.hist_visual)

        self._update_axes()
        self.canvas.update()
        self.update_status()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(AmplitudeView, self).attach(gui)

        # Amplitude type actions.
        def _make_amplitude_action(a):
            def callback():
                self.amplitudes_type = a
                self.plot()
            return callback

        for a in self.amplitudes_types.keys():
            name = 'Change amplitudes type to %s' % a
            self.actions.add(
                _make_amplitude_action(a), show_shortcut=False,
                name=name, view_submenu='Change amplitudes type')

        self.actions.add(self.next_amplitudes_type, set_busy=True)
        self.actions.add(self.previous_amplitudes_type, set_busy=True)

    @property
    def status(self):
        return self.amplitudes_type

    @property
    def amplitudes_type(self):
        return self.amplitudes_types.current

    @amplitudes_type.setter
    def amplitudes_type(self, value):
        self.amplitudes_types.set(value)

    def next_amplitudes_type(self):
        """Switch to the next amplitudes type."""
        self.amplitudes_types.next()
        logger.debug("Switch to amplitudes type: %s.", self.amplitudes_types.current)
        self.plot()

    def previous_amplitudes_type(self):
        """Switch to the previous amplitudes type."""
        self.amplitudes_types.previous()
        logger.debug("Switch to amplitudes type: %s.", self.amplitudes_types.current)
        self.plot()

    def on_mouse_click(self, e):
        """Select a time from the amplitude view to display in the trace view."""
        if 'Alt' in e.modifiers:
            mouse_pos = self.canvas.panzoom.window_to_ndc(e.pos)
            time = Range(NDC, self.data_bounds).apply(mouse_pos)[0][0]
            emit('select_time', self, time)
