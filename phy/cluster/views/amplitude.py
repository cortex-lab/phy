"""Amplitude view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from phylib.utils._types import _as_array
from phylib.utils.event import connect, emit, unconnect

from phy.cluster._utils import RotatingProperty
from phy.plot.transform import NDC, Range, Rotate, Scale, Translate
from phy.plot.visuals import HistogramVisual, LineVisual, PatchVisual, ScatterVisual
from phy.utils.color import add_alpha, selected_cluster_color

from .base import LassoMixin, ManualClusteringView, MarkerSizeMixin, RecordingTimeAxisMixin
from .histogram import _compute_histogram

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Amplitude view
# -----------------------------------------------------------------------------


class AmplitudeView(RecordingTimeAxisMixin, MarkerSizeMixin, LassoMixin, ManualClusteringView):
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
    defer_hidden_updates = True

    _default_position = 'right'

    # Alpha channel of the markers in the scatter plot.
    marker_alpha = 1.0
    time_range_color = (1.0, 1.0, 0.0, 0.25)
    split_preview_color = (1.0, 0.4, 0.1, 1.0)

    # Number of bins in the histogram.
    n_bins = 100

    # Alpha channel of the histogram in the background.
    histogram_alpha = 0.5

    # Quantile used for scaling of the amplitudes (less than 1 to avoid outliers).
    quantile = 0.99

    # Size of the histogram, between 0 and 1.
    histogram_scale = 0.25

    default_shortcuts = {
        'change_marker_size': 'alt+wheel',
        'next_amplitudes_type': 'a',
        'previous_amplitudes_type': 'shift+a',
        'select_x_dim': 'shift+left click',
        'select_y_dim': 'shift+right click',
        'select_time': 'alt+click',
    }

    def __init__(
        self, amplitudes=None, amplitudes_type=None, duration=None, split_is_eligible=None
    ):
        super().__init__()
        self.state_attrs += ('amplitudes_type',)

        # The split preview is deliberately transient and is not part of view state.
        self.split_threshold = None
        self._split_is_eligible = split_is_eligible
        self._split_threshold_dragging = False
        self._displayed_bunchs = ()

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
        self.duration = duration or 1.0

        # Histogram visual.
        self.hist_visual = HistogramVisual()
        self.hist_visual.transforms.add(
            [
                Range(NDC, (-1, -1, 1, -1 + 2 * self.histogram_scale)),
                Rotate('cw'),
                Scale((1, -1)),
                Translate((2.05, 0)),
            ]
        )
        self.canvas.add_visual(self.hist_visual)
        self.canvas.panzoom.zoom = self.canvas.panzoom._default_zoom = (0.75, 1)
        self.canvas.panzoom.pan = self.canvas.panzoom._default_pan = (-0.25, 0)

        # Yellow vertical bar showing the selected time interval.
        self.patch_visual = PatchVisual(primitive_type='triangle_fan')
        self.patch_visual.inserter.insert_vert(
            """
            const float MIN_INTERVAL_SIZE = 0.01;
            uniform float u_interval_size;
        """,
            'header',
        )
        self.patch_visual.inserter.insert_vert(
            """
            gl_Position.y = pos_orig.y;

            // The following is used to ensure that (1) the bar width increases with the zoom level
            // but also (2) there is a minimum absolute width so that the bar remains visible
            // at low zoom levels.
            float w = max(MIN_INTERVAL_SIZE, u_interval_size * u_zoom.x);
            // HACK: the z coordinate is used to store 0 or 1, depending on whether the current
            // vertex is on the left or right edge of the bar.
            gl_Position.x += w * (-1 + 2 * int(a_position.z == 0));

        """,
            'after_transforms',
        )
        self.canvas.add_visual(self.patch_visual)

        # Horizontal amplitude split threshold, expressed in amplitude data coordinates.
        self.split_threshold_visual = LineVisual()
        self.split_threshold_visual.hide()
        self.canvas.add_visual(self.split_threshold_visual)

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
            for bunch in bunchs
            if len(bunch.amplitudes)
        )
        m = min(0, m)  # ensure ymin <= 0
        M = max(
            np.quantile(bunch.amplitudes, self.quantile)
            for bunch in bunchs
            if len(bunch.amplitudes)
        )
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
        xm = 0.5 * (x0 + x1)
        pos = np.array(
            [
                [xm, -1],
                [xm, +1],
                [xm, +1],
                [xm, -1],
            ]
        )
        self.patch_visual.program['u_interval_size'] = 0.5 * (x1 - x0)
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
            color=add_alpha(bunch.color, self.histogram_alpha),
        )

        # Scatter plot.
        color = bunch.color
        if bunch.cluster_id is not None and self.split_threshold is not None:
            color = np.tile(np.asarray(bunch.color), (len(bunch.amplitudes), 1))
            below = np.isfinite(bunch.amplitudes) & (bunch.amplitudes < self.split_threshold)
            color[below] = self.split_preview_color
        self.visual.add_batch_data(
            pos=bunch.pos, color=color, size=ms, data_bounds=self.data_bounds
        )

    def _update_split_threshold_visual(self):
        if self.split_threshold is None or not hasattr(self, 'data_bounds'):
            self.split_threshold_visual.hide()
            return
        xmin, _, xmax, _ = self.data_bounds
        y = self.split_threshold
        self.split_threshold_visual.set_data(
            pos=np.array([[xmin, y, xmax, y]]),
            color=self.split_preview_color,
            data_bounds=self.data_bounds,
        )
        self.split_threshold_visual.show()

    def _replot_displayed_amplitudes(self):
        """Recolor the already loaded amplitude sample without loading data."""
        if not self._displayed_bunchs:
            self._update_split_threshold_visual()
            self.canvas.update()
            return
        self.visual.reset_batch()
        for bunch in self._displayed_bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.visual)
        self._update_split_threshold_visual()
        self.canvas.update()

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
            bunch.spike_ids = spike_ids
            bunch.spike_times = spike_times
            bunch.amplitudes = amplitudes
            # Ensure that bunch.pos exists, as it used by the LassoMixin.
            bunch.pos = np.c_[spike_times, amplitudes]
            assert bunch.pos.ndim == 2
            bunch.cluster_id = cluster_id
            bunch.color = (
                selected_cluster_color(
                    self.cluster_color_index(cluster_id, i - 1), self.marker_alpha
                )
                # Background amplitude color.
                if cluster_id is not None
                else (0.5, 0.5, 0.5, 0.5)
            )
        return bunchs

    def plot(self, **kwargs):
        """Update the view with the current cluster selection."""
        bunchs = self.get_clusters_data(**kwargs)
        if not bunchs:
            return
        self.data_bounds = self._get_data_bounds(bunchs)
        bunchs = self._add_histograms(bunchs)
        self._displayed_bunchs = tuple(bunchs)
        # Use the same scale for all histograms.
        self._ylim = max(bunch.histogram.max() for bunch in bunchs) if bunchs else 1.0

        self.visual.reset_batch()
        self.hist_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.hist_visual)
        self._update_split_threshold_visual()

        self._update_axes()
        self.canvas.update()
        self.update_status()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super().attach(gui)

        # Amplitude type actions.
        def _make_amplitude_action(a):
            def callback():
                self.amplitudes_type = a
                self.plot()

            return callback

        for a in self.amplitudes_types.keys():
            name = f'Change amplitudes type to {a}'
            self.actions.add(
                _make_amplitude_action(a),
                show_shortcut=False,
                name=name,
                view_submenu='Change amplitudes type',
            )

        self.actions.add(self.next_amplitudes_type, set_busy=True)
        self.actions.add(self.previous_amplitudes_type, set_busy=True)
        self.actions.add(self.clear_amplitude_split_threshold, show_shortcut=False)

        @connect(event='lasso_updated', sender=self.canvas)
        def on_lasso_updated(sender, polygon):
            if len(polygon):
                self.clear_amplitude_split_threshold()

        @connect(event='close_view', sender=self)
        def on_close_view(view, gui):
            unconnect(on_lasso_updated)
            unconnect(on_close_view)

    @property
    def status(self):
        return self.amplitudes_type

    @property
    def amplitudes_type(self):
        return self.amplitudes_types.current

    @amplitudes_type.setter
    def amplitudes_type(self, value):
        if hasattr(self, 'split_threshold') and value != self.amplitudes_types.current:
            self.clear_amplitude_split_threshold()
        self.amplitudes_types.set(value)

    def next_amplitudes_type(self):
        """Switch to the next amplitudes type."""
        self.clear_amplitude_split_threshold()
        self.amplitudes_types.next()
        logger.debug('Switch to amplitudes type: %s.', self.amplitudes_types.current)
        self.plot()

    def previous_amplitudes_type(self):
        """Switch to the previous amplitudes type."""
        self.clear_amplitude_split_threshold()
        self.amplitudes_types.previous()
        logger.debug('Switch to amplitudes type: %s.', self.amplitudes_types.current)
        self.plot()

    def on_mouse_click(self, e):
        """Select a time from the amplitude view to display in the trace view."""
        if 'Control' in e.modifiers and e.button == 'Right':
            self.clear_split_selection()
        elif 'Alt' in e.modifiers and e.button == 'Left':
            mouse_pos = self.canvas.panzoom.window_to_ndc(e.pos)
            time = Range(NDC, self.data_bounds).apply(mouse_pos)[0][0]
            emit('select_time', self, time)

    def _can_set_split_threshold(self):
        eligible = len(self.cluster_ids) == 1
        if eligible and self._split_is_eligible is not None:
            eligible = bool(self._split_is_eligible())
        if not eligible:
            self._show_split_status(
                'Amplitude threshold splitting requires exactly one selected cluster '
                'and inactive Merge mode.'
            )
        return eligible

    def _show_split_status(self, message):
        logger.warning(message)
        if hasattr(self, 'dock'):
            self.dock.set_status(message)

    def _threshold_from_window_pos(self, pos):
        mouse_pos = self.canvas.panzoom.window_to_ndc(pos)
        return float(Range(NDC, self.data_bounds).apply(mouse_pos)[0][1])

    def _set_split_threshold_from_pos(self, pos):
        self.split_threshold = self._threshold_from_window_pos(pos)
        self.activate_split_selection()
        self._replot_displayed_amplitudes()
        emit(
            'amplitude_split_preview_changed',
            self,
            cluster_id=self.cluster_ids[0],
            amplitudes_type=self.amplitudes_type,
            threshold=self.split_threshold,
        )

    def on_mouse_press(self, e):
        if e.button != 'Right' or 'Alt' not in e.modifiers or not self._can_set_split_threshold():
            return
        self.canvas.lasso.clear()
        self._split_threshold_dragging = True
        self._set_split_threshold_from_pos(e.pos)

    def on_mouse_move(self, e):
        if not self._split_threshold_dragging:
            return
        if e.button != 'Right' or 'Alt' not in (e.mouse_press_modifiers or ()):
            return
        self._set_split_threshold_from_pos(e.pos)

    def on_mouse_release(self, e):
        if not self._split_threshold_dragging:
            return
        self._split_threshold_dragging = False
        if e.button == 'Right' and 'Alt' in e.modifiers:
            self._set_split_threshold_from_pos(e.pos)

    def clear_amplitude_split_threshold(self):
        """Clear the amplitude split threshold."""
        if self.split_threshold is None:
            return
        cluster_id = self.cluster_ids[0] if len(self.cluster_ids) == 1 else None
        self.split_threshold = None
        self._split_threshold_dragging = False
        self._replot_displayed_amplitudes()
        emit(
            'amplitude_split_preview_changed',
            self,
            cluster_id=cluster_id,
            amplitudes_type=self.amplitudes_type,
            threshold=None,
        )

    def clear_split_selection(self):
        super().clear_split_selection()
        self.clear_amplitude_split_threshold()

    def on_select(self, cluster_ids=None, **kwargs):
        self.clear_amplitude_split_threshold()
        super().on_select(cluster_ids=cluster_ids, **kwargs)

    def on_cluster(self, up):
        self.clear_amplitude_split_threshold()

    def on_request_split(self, sender=None):
        if self.split_threshold is None:
            return super().on_request_split(sender=sender)
        if len(self.cluster_ids) != 1:
            return np.array([], dtype=np.int64)

        bunchs = self.get_clusters_data(load_all=True) or ()
        if len(bunchs) != 1:
            self._show_split_status('Amplitude threshold split has no eligible spikes.')
            return np.array([], dtype=np.int64)
        bunch = bunchs[0]
        spike_ids = _as_array(bunch.spike_ids)
        amplitudes = _as_array(bunch.amplitudes)
        selected = np.isfinite(amplitudes) & (amplitudes < self.split_threshold)
        n_selected = int(selected.sum())
        if n_selected == 0:
            self._show_split_status(
                'Amplitude threshold split rejected: no spikes are below the threshold.'
            )
            return np.array([], dtype=np.int64)
        if n_selected == len(spike_ids):
            self._show_split_status(
                'Amplitude threshold split rejected: all spikes are below the threshold.'
            )
            return np.array([], dtype=np.int64)
        out = np.unique(spike_ids[selected]).astype(np.int64, copy=False)
        self.clear_split_selection()
        return out
