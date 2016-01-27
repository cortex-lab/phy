# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from vispy.util.event import Event

from phy.io.array import _index_of, _get_padded, get_excerpts
from phy.gui import Actions
from phy.plot import View, _get_linear_x
from phy.plot.utils import _get_boxes
from phy.stats import correlograms
from phy.utils import Bunch

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

# Default color map for the selected clusters.
_COLORMAP = np.array([[8, 146, 252],
                      [255, 2, 2],
                      [240, 253, 2],
                      [228, 31, 228],
                      [2, 217, 2],
                      [255, 147, 2],
                      [212, 150, 70],
                      [205, 131, 201],
                      [201, 172, 36],
                      [150, 179, 62],
                      [95, 188, 122],
                      [129, 173, 190],
                      [231, 107, 119],
                      ])


def _selected_clusters_colors(n_clusters=None):
    if n_clusters is None:
        n_clusters = _COLORMAP.shape[0]
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.


def _extract_wave(traces, start, mask, wave_len=None, mask_threshold=.5):
    n_samples, n_channels = traces.shape
    assert mask.shape == (n_channels,)
    channels = np.nonzero(mask > mask_threshold)[0]
    # There should be at least one non-masked channel.
    if not len(channels):
        return  # pragma: no cover
    i, j = start, start + wave_len
    a, b = max(0, i), min(j, n_samples - 1)
    data = traces[a:b, channels]
    data = _get_padded(data, i - a, i - a + wave_len)
    assert data.shape == (wave_len, len(channels))
    return data, channels


def _get_depth(masks, spike_clusters_rel=None, n_clusters=None):
    """Return the OpenGL z-depth of vertices as a function of the
    mask and cluster index."""
    n_spikes = len(masks)
    assert masks.shape == (n_spikes,)
    # Fixed depth for background spikes.
    if spike_clusters_rel is None:
        depth = .5 * np.ones(n_spikes)
    else:
        depth = (-0.1 - (spike_clusters_rel + masks) /
                 float(n_clusters + 10.))
    depth[masks <= 0.25] = 0
    assert depth.shape == (n_spikes,)
    return depth


def _get_color(masks, spike_clusters_rel=None, n_clusters=None, alpha=.5):
    """Return the color of vertices as a function of the mask and
    cluster index."""
    n_spikes = masks.shape[0]
    # The transparency depends on whether the spike clusters are specified.
    # For background spikes, we use a smaller alpha.
    alpha = alpha if spike_clusters_rel is not None else .25
    assert masks.shape == (n_spikes,)
    # Generate the colors.
    colors = _selected_clusters_colors(n_clusters)
    # Color as a function of the mask.
    if spike_clusters_rel is not None:
        assert spike_clusters_rel.shape == (n_spikes,)
        color = colors[spike_clusters_rel]
    else:
        # Fixed color when the spike clusters are not specified.
        color = np.ones((n_spikes, 3))
    hsv = rgb_to_hsv(color[:, :3])
    # Change the saturation and value as a function of the mask.
    hsv[:, 1] *= masks
    hsv[:, 2] *= .5 * (1. + masks)
    color = hsv_to_rgb(hsv)
    color = np.c_[color, alpha * np.ones((n_spikes, 1))]
    return color


def _extend(channels, n=None):
    channels = list(channels)
    if n is None:
        return channels
    if len(channels) < n:
        channels.extend([channels[-1]] * (n - len(channels)))
    channels = channels[:n]
    assert len(channels) == n
    return channels


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

class StatusEvent(Event):
    def __init__(self, type, message=None):
        super(StatusEvent, self).__init__(type)
        self.message = message


class ManualClusteringView(View):
    """Base class for clustering views.

    The views take their data with functions `cluster_ids: spike_ids, data`.

    """
    default_shortcuts = {
    }

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Message to show in the status bar.
        self.status = None

        # Attached GUI.
        self.gui = None

        # Keep track of the selected clusters and spikes.
        self.cluster_ids = None

        super(ManualClusteringView, self).__init__(**kwargs)
        self.events.add(status=StatusEvent)

    def on_select(self, cluster_ids=None):
        cluster_ids = (cluster_ids if cluster_ids is not None
                       else self.cluster_ids)
        self.cluster_ids = list(cluster_ids) if cluster_ids is not None else []
        self.cluster_ids = [int(c) for c in self.cluster_ids]

    def attach(self, gui):
        """Attach the view to the GUI."""

        # Disable keyboard pan so that we can use arrows as global shortcuts
        # in the GUI.
        self.panzoom.enable_keyboard_pan = False

        gui.add_view(self)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self))

        gui.connect_(self.on_select)
        self.actions = Actions(gui,
                               name=self.__class__.__name__,
                               menu=self.__class__.__name__,
                               default_shortcuts=self.shortcuts)

        # Update the GUI status message when the `self.set_status()` method
        # is called, i.e. when the `status` event is raised by the VisPy
        # view.
        @self.connect
        def on_status(e):
            gui.status_message = e.message

        # Save the view state in the GUI state.
        @gui.connect_
        def on_close():
            gui.state.update_view_state(self, self.state)
            # NOTE: create_gui() already saves the state, but the event
            # is registered *before* we add all views.
            gui.state.save()

        self.show()

    @property
    def state(self):
        """View state.

        This Bunch will be automatically persisted in the GUI state when the
        GUI is closed.

        To be overriden.

        """
        return Bunch()

    def set_state(self, state):
        """Set the view state.

        The passed object is the persisted `self.state` bunch.

        May be overriden.

        """
        for k, v in state.items():
            setattr(self, k, v)

    def set_status(self, message=None):
        message = message or self.status
        if not message:
            return
        self.status = message
        self.events.status(message=message)

    def on_mouse_move(self, e):  # pragma: no cover
        self.set_status()


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

class ChannelClick(Event):
    def __init__(self, type, channel_idx=None, key=None, button=None):
        super(ChannelClick, self).__init__(type)
        self.channel_idx = channel_idx
        self.key = key
        self.button = button


class WaveformView(ManualClusteringView):
    scaling_coeff = 1.1

    default_shortcuts = {
        'toggle_waveform_overlap': 'o',
        'toggle_zoom_on_channels': 'z',

        # Box scaling.
        'widen': 'ctrl+right',
        'narrow': 'ctrl+left',
        'increase': 'ctrl+up',
        'decrease': 'ctrl+down',

        # Probe scaling.
        'extend_horizontally': 'shift+right',
        'shrink_horizontally': 'shift+left',
        'extend_vertically': 'shift+up',
        'shrink_vertically': 'shift+down',
    }

    def __init__(self,
                 waveforms=None,
                 channel_positions=None,
                 n_samples=None,
                 waveform_lim=None,
                 best_channels=None,
                 **kwargs):
        self._key_pressed = None
        self._overlap = False
        self.do_zoom_on_channels = True

        self.best_channels = best_channels or (lambda clusters, n=None: [])

        # Channel positions and n_channels.
        assert channel_positions is not None
        self.channel_positions = np.asarray(channel_positions)
        self.n_channels = self.channel_positions.shape[0]

        # Number of samples per waveform.
        n_samples = (sum(map(abs, n_samples)) if isinstance(n_samples, tuple)
                     else n_samples)
        assert n_samples > 0
        self.n_samples = n_samples

        # Initialize the view.
        box_bounds = _get_boxes(channel_positions)
        super(WaveformView, self).__init__(layout='boxed',
                                           box_bounds=box_bounds,
                                           **kwargs)

        self.events.add(channel_click=ChannelClick)

        # Box and probe scaling.
        self._box_scaling = np.ones(2)
        self._probe_scaling = np.ones(2)

        # Make a copy of the initial box pos and size. We'll apply the scaling
        # to these quantities.
        self.box_pos = np.array(self.boxed.box_pos)
        self.box_size = np.array(self.boxed.box_size)
        self._update_boxes()

        # Data: functions cluster_id => waveforms.
        self.waveforms = waveforms

        # Waveform normalization.
        assert waveform_lim > 0
        self.data_bounds = [-1, -waveform_lim, +1, +waveform_lim]

        # Channel positions.
        assert channel_positions.shape == (self.n_channels, 2)
        self.channel_positions = channel_positions

    def _get_data(self, cluster_ids):
        d = self.waveforms(cluster_ids)
        d.alpha = .5
        return d

    def on_select(self, cluster_ids=None):
        super(WaveformView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Load the waveform subset.
        data = self._get_data(cluster_ids)
        alpha = data.alpha
        spike_ids = data.spike_ids
        spike_clusters = data.spike_clusters
        w = data.waveforms
        masks = data.masks
        n_spikes = len(spike_ids)
        assert w.shape == (n_spikes, self.n_samples, self.n_channels)
        assert masks.shape == (n_spikes, self.n_channels)

        # Relative spike clusters.
        spike_clusters_rel = _index_of(spike_clusters, cluster_ids)
        assert spike_clusters_rel.shape == (n_spikes,)

        # Fetch the waveforms.
        t = _get_linear_x(n_spikes, self.n_samples)
        # Overlap.
        if not self.overlap:
            t = t + 2.5 * (spike_clusters_rel[:, np.newaxis] -
                           (n_clusters - 1) / 2.)
            # The total width should not depend on the number of clusters.
            t /= n_clusters

        # Plot all waveforms.
        # OPTIM: avoid the loop.
        with self.building():
            for ch in range(self.n_channels):
                m = masks[:, ch]
                depth = _get_depth(m,
                                   spike_clusters_rel=spike_clusters_rel,
                                   n_clusters=n_clusters)
                color = _get_color(m,
                                   spike_clusters_rel=spike_clusters_rel,
                                   n_clusters=n_clusters,
                                   alpha=alpha,
                                   )
                self[ch].plot(x=t, y=w[:, :, ch],
                              color=color,
                              depth=depth,
                              data_bounds=self.data_bounds,
                              )

        # Zoom on the best channels when selecting clusters.
        channels = self.best_channels(cluster_ids)
        if channels is not None and self.do_zoom_on_channels:
            self.zoom_on_channels(channels)

    @property
    def state(self):
        return Bunch(box_scaling=tuple(self.box_scaling),
                     probe_scaling=tuple(self.probe_scaling),
                     overlap=self.overlap,
                     do_zoom_on_channels=self.do_zoom_on_channels,
                     )

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(WaveformView, self).attach(gui)
        self.actions.add(self.toggle_waveform_overlap)
        self.actions.add(self.toggle_zoom_on_channels)

        # Box scaling.
        self.actions.add(self.widen)
        self.actions.add(self.narrow)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)

        # Probe scaling.
        self.actions.add(self.extend_horizontally)
        self.actions.add(self.shrink_horizontally)
        self.actions.add(self.extend_vertically)
        self.actions.add(self.shrink_vertically)

        # We forward the event from VisPy to the phy GUI.
        @self.connect
        def on_channel_click(e):
            gui.emit('channel_click',
                     channel_idx=e.channel_idx,
                     key=e.key,
                     button=e.button,
                     )

    # Overlap
    # -------------------------------------------------------------------------

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value
        # HACK: temporarily disable automatic zoom on channels when
        # changing the overlap.
        tmp = self.do_zoom_on_channels
        self.do_zoom_on_channels = False
        self.on_select()
        self.do_zoom_on_channels = tmp

    def toggle_waveform_overlap(self):
        """Toggle the overlap of the waveforms."""
        self.overlap = not self.overlap

    # Box scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.boxed.update_boxes(self.box_pos * self.probe_scaling,
                                self.box_size * self.box_scaling)

    @property
    def box_scaling(self):
        return self._box_scaling

    @box_scaling.setter
    def box_scaling(self, value):
        assert len(value) == 2
        self._box_scaling = np.array(value)
        self._update_boxes()

    def widen(self):
        """Increase the horizontal scaling of the waveforms."""
        self._box_scaling[0] *= self.scaling_coeff
        self._update_boxes()

    def narrow(self):
        """Decrease the horizontal scaling of the waveforms."""
        self._box_scaling[0] /= self.scaling_coeff
        self._update_boxes()

    def increase(self):
        """Increase the vertical scaling of the waveforms."""
        self._box_scaling[1] *= self.scaling_coeff
        self._update_boxes()

    def decrease(self):
        """Decrease the vertical scaling of the waveforms."""
        self._box_scaling[1] /= self.scaling_coeff
        self._update_boxes()

    # Probe scaling
    # -------------------------------------------------------------------------

    @property
    def probe_scaling(self):
        return self._probe_scaling

    @probe_scaling.setter
    def probe_scaling(self, value):
        assert len(value) == 2
        self._probe_scaling = np.array(value)
        self._update_boxes()

    def extend_horizontally(self):
        """Increase the horizontal scaling of the probe."""
        self._probe_scaling[0] *= self.scaling_coeff
        self._update_boxes()

    def shrink_horizontally(self):
        """Decrease the horizontal scaling of the waveforms."""
        self._probe_scaling[0] /= self.scaling_coeff
        self._update_boxes()

    def extend_vertically(self):
        """Increase the vertical scaling of the waveforms."""
        self._probe_scaling[1] *= self.scaling_coeff
        self._update_boxes()

    def shrink_vertically(self):
        """Decrease the vertical scaling of the waveforms."""
        self._probe_scaling[1] /= self.scaling_coeff
        self._update_boxes()

    # Navigation
    # -------------------------------------------------------------------------

    def toggle_zoom_on_channels(self):
        self.do_zoom_on_channels = not self.do_zoom_on_channels

    def zoom_on_channels(self, channels_rel):
        """Zoom on some channels."""
        if channels_rel is None or not len(channels_rel):
            return
        channels_rel = np.asarray(channels_rel, dtype=np.int32)
        assert 0 <= channels_rel.min() <= channels_rel.max() < self.n_channels
        # Bounds of the channels.
        b = self.boxed.box_bounds[channels_rel]
        x0, y0 = b[:, :2].min(axis=0)
        x1, y1 = b[:, 2:].max(axis=0)
        self.panzoom.set_range((x0, y0, x1, y1), keep_aspect=True)

    def on_key_press(self, event):
        """Handle key press events."""
        key = event.key
        self._key_pressed = key

    def on_mouse_press(self, e):
        key = self._key_pressed
        if 'Control' in e.modifiers or key in map(str, range(10)):
            key = int(key.name) if key in map(str, range(10)) else None
            # Get mouse position in NDC.
            mouse_pos = self.panzoom.get_mouse_pos(e.pos)
            channel_idx = self.boxed.get_closest_box(mouse_pos)
            self.events.channel_click(channel_idx=channel_idx,
                                      key=key,
                                      button=e.button)

    def on_key_release(self, event):
        self._key_pressed = None


# -----------------------------------------------------------------------------
# Trace view
# -----------------------------------------------------------------------------

def select_traces(traces, interval, sample_rate=None):
    """Load traces in an interval (in seconds)."""
    start, end = interval
    i, j = round(sample_rate * start), round(sample_rate * end)
    i, j = int(i), int(j)
    traces = traces[i:j, :]
    return traces


def extract_spikes(traces, interval, sample_rate=None,
                   spike_times=None, spike_clusters=None,
                   all_masks=None,
                   n_samples_waveforms=None):
    sr = sample_rate
    ns = n_samples_waveforms
    if not isinstance(ns, tuple):
        ns = (ns // 2, ns // 2)
    offset_samples = ns[0]
    wave_len = ns[0] + ns[1]

    # Find spikes.
    a, b = spike_times.searchsorted(interval)
    st = spike_times[a:b]
    sc = spike_clusters[a:b]
    m = all_masks[a:b]
    n = len(st)
    assert len(sc) == n
    assert m.shape[0] == n

    # Extract waveforms.
    spikes = []
    for i in range(n):
        b = Bunch()
        # Find the start of the waveform in the extracted traces.
        sample_start = int(round((st[i] - interval[0]) * sr))
        sample_start -= offset_samples
        b.waveforms, b.channels = _extract_wave(traces,
                                                sample_start,
                                                m[i],
                                                wave_len)
        # Masks on unmasked channels.
        b.masks = m[i, b.channels]
        b.spike_time = st[i]
        b.spike_cluster = sc[i]
        b.offset_samples = offset_samples

        spikes.append(b)
    return spikes


class TraceView(ManualClusteringView):
    interval_duration = .5  # default duration of the interval
    shift_amount = .1
    scaling_coeff = 1.1
    default_trace_color = (.3, .3, .3, 1.)
    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
        'decrease': 'alt+down',
        'increase': 'alt+up',
        'widen': 'ctrl+alt+left',
        'narrow': 'ctrl+alt+right',
    }

    def __init__(self,
                 traces=None,
                 spikes=None,
                 sample_rate=None,
                 duration=None,
                 n_channels=None,
                 **kwargs):

        # traces is a function interval => [traces]
        # spikes is a function interval => [Bunch(...)]

        # Sample rate.
        assert sample_rate > 0
        self.sample_rate = sample_rate
        self.dt = 1. / self.sample_rate

        # Traces and spikes.
        assert hasattr(traces, '__call__')
        self.traces = traces
        assert hasattr(spikes, '__call__')
        self.spikes = spikes

        assert duration >= 0
        self.duration = duration

        assert n_channels >= 0
        self.n_channels = n_channels

        # Box and probe scaling.
        self._scaling = 1.
        self._origin = None

        # Initialize the view.
        super(TraceView, self).__init__(layout='stacked',
                                        origin=self.origin,
                                        n_plots=self.n_channels,
                                        **kwargs)

        # Make a copy of the initial box pos and size. We'll apply the scaling
        # to these quantities.
        self.box_size = np.array(self.stacked.box_size)
        self._update_boxes()

        # Initial interval.
        self.interval = None
        self.go_to(duration / 2.)

    # Internal methods
    # -------------------------------------------------------------------------

    def _plot_traces(self, traces=None, color=None):
        assert traces.shape[1] == self.n_channels
        t = self.interval[0] + np.arange(traces.shape[0]) * self.dt
        t = np.tile(t, (self.n_channels, 1))
        color = color or self.default_trace_color
        channels = np.arange(self.n_channels)
        for ch in channels:
            self[ch].plot(t[ch, :], traces[:, ch],
                          color=color,
                          data_bounds=self.data_bounds,
                          )

    def _plot_spike(self, waveforms=None, channels=None, masks=None,
                    spike_time=None, spike_cluster=None, offset_samples=0,
                    color=None):

        n_samples, n_channels = waveforms.shape
        assert len(channels) == n_channels
        assert len(masks) == n_channels
        sr = float(self.sample_rate)

        t0 = spike_time - offset_samples / sr

        # Determine the color as a function of the spike's cluster.
        if color is None:
            clu = spike_cluster
            if self.cluster_ids is None or clu not in self.cluster_ids:
                sc = None
                n_clusters = None
            else:
                clu_rel = self.cluster_ids.index(clu)
                sc = clu_rel * np.ones(n_channels, dtype=np.int32)
                n_clusters = len(self.cluster_ids)
            color = _get_color(masks,
                               spike_clusters_rel=sc,
                               n_clusters=n_clusters)

        # Generate the x coordinates of the waveform.
        t = t0 + self.dt * np.arange(n_samples)
        t = np.tile(t, (n_channels, 1))  # (n_unmasked_channels, n_samples)

        # The box index depends on the channel.
        box_index = np.repeat(channels[:, np.newaxis], n_samples, axis=0)
        self.plot(t, waveforms.T, color=color, box_index=box_index,
                  data_bounds=self.data_bounds)

    def _restrict_interval(self, interval):
        start, end = interval
        # Restrict the interval to the boundaries of the traces.
        if start < 0:
            end += (-start)
            start = 0
        elif end >= self.duration:
            start -= (end - self.duration)
            end = self.duration
        start = np.clip(start, 0, end)
        end = np.clip(end, start, self.duration)
        assert 0 <= start < end <= self.duration
        return start, end

    # Public methods
    # -------------------------------------------------------------------------

    def set_interval(self, interval, change_status=True):
        """Display the traces and spikes in a given interval."""
        self.clear()
        interval = self._restrict_interval(interval)
        self.interval = interval
        start, end = interval
        # Set the status message.
        if change_status:
            self.set_status('Interval: {:.3f} s - {:.3f} s'.format(start, end))

        # Load the traces.
        all_traces = self.traces(interval)
        assert isinstance(all_traces, (tuple, list))
        # Default data bounds.
        m, M = all_traces[0].traces.min(), all_traces[0].traces.max()
        self.data_bounds = np.array([start, m, end, M])

        # Plot the traces.
        for traces in all_traces:
            self._plot_traces(**traces)

        # Plot the spikes.
        spikes = self.spikes(interval, all_traces)
        assert isinstance(spikes, (tuple, list))
        for spike in spikes:
            self._plot_spike(**spike)

        self.build()
        self.update()

    def on_select(self, cluster_ids=None):
        super(TraceView, self).on_select(cluster_ids)
        self.set_interval(self.interval, change_status=False)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(TraceView, self).attach(gui)
        self.actions.add(self.go_to, alias='tg')
        self.actions.add(self.shift, alias='ts')
        self.actions.add(self.go_right)
        self.actions.add(self.go_left)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.add(self.widen)
        self.actions.add(self.narrow)

    @property
    def state(self):
        return Bunch(scaling=self.scaling,
                     origin=self.origin,
                     )

    # Scaling
    # -------------------------------------------------------------------------

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value
        self._update_boxes()

    # Origin
    # -------------------------------------------------------------------------

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value
        self._update_boxes()

    # Navigation
    # -------------------------------------------------------------------------

    @property
    def time(self):
        """Time at the center of the window."""
        return sum(self.interval) * .5

    @property
    def half_duration(self):
        """Half of the duration of the current interval."""
        if self.interval is not None:
            a, b = self.interval
            return (b - a) * .5
        else:
            return self.interval_duration * .5

    def go_to(self, time):
        """Go to a specific time (in seconds)."""
        half_dur = self.half_duration
        self.set_interval((time - half_dur, time + half_dur))

    def shift(self, delay):
        """Shift the interval by a given delay (in seconds)."""
        self.go_to(self.time + delay)

    def go_right(self):
        """Go to right."""
        start, end = self.interval
        delay = (end - start) * .2
        self.shift(delay)

    def go_left(self):
        """Go to left."""
        start, end = self.interval
        delay = (end - start) * .2
        self.shift(-delay)

    def widen(self):
        """Increase the interval size."""
        t, h = self.time, self.half_duration
        h *= self.scaling_coeff
        self.set_interval((t - h, t + h))

    def narrow(self):
        """Decrease the interval size."""
        t, h = self.time, self.half_duration
        h /= self.scaling_coeff
        self.set_interval((t - h, t + h))

    # Channel scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.stacked.box_size = self.box_size * self.scaling

    def increase(self):
        """Increase the scaling of the traces."""
        self.scaling *= self.scaling_coeff
        self._update_boxes()

    def decrease(self):
        """Decrease the scaling of the traces."""
        self.scaling /= self.scaling_coeff
        self._update_boxes()


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------

def _dimensions_matrix(x_channels, y_channels, n_cols=None,
                       top_left_attribute=None):
    """Dimensions matrix."""
    # time, attr      time,    (x, 0)     time,    (y, 0)     time, (z, 0)
    # time, (x', 0)   (x', 0), (x, 0)     (x', 1), (y, 0)     (x', 2), (z, 0)
    # time, (y', 0)   (y', 0), (x, 1)     (y', 1), (y, 1)     (y', 2), (z, 1)
    # time, (z', 0)   (z', 0), (x, 2)     (z', 1), (y, 2)     (z', 2), (z, 2)

    assert n_cols > 0
    assert len(x_channels) >= n_cols - 1
    assert len(y_channels) >= n_cols - 1

    y_dim = {}
    x_dim = {}
    x_dim[0, 0] = 'time'
    y_dim[0, 0] = top_left_attribute or 'time'

    # Time in first column and first row.
    for i in range(1, n_cols):
        x_dim[0, i] = 'time'
        y_dim[0, i] = (x_channels[i - 1], 0)
        x_dim[i, 0] = 'time'
        y_dim[i, 0] = (y_channels[i - 1], 0)

    for i in range(1, n_cols):
        for j in range(1, n_cols):
            x_dim[i, j] = (x_channels[i - 1], j - 1)
            y_dim[i, j] = (y_channels[j - 1], i - 1)

    return x_dim, y_dim


def _project_mask_depth(dim, masks, spike_clusters_rel=None, n_clusters=None):
    """Return the mask and depth vectors for a given dimension."""
    n_spikes = masks.shape[0]
    if isinstance(dim, tuple):
        ch, fet = dim
        m = masks[:, ch]
        d = _get_depth(m,
                       spike_clusters_rel=spike_clusters_rel,
                       n_clusters=n_clusters)
    else:
        m = np.ones(n_spikes)
        d = np.zeros(n_spikes)
    return m, d


class FeatureView(ManualClusteringView):
    _default_marker_size = 3.

    default_shortcuts = {
        'increase': 'ctrl++',
        'decrease': 'ctrl+-',
    }

    def __init__(self,
                 features=None,
                 background_features=None,
                 spike_times=None,
                 n_channels=None,
                 n_features_per_channel=None,
                 feature_lim=None,
                 best_channels=None,
                 **kwargs):
        """
        features is a function :
            `cluster_ids: Bunch(spike_ids,
                                features,
                                masks,
                                spike_clusters,
                                spike_times)`
        background_features is a Bunch(...) like above.

        """
        self._scaling = 1.

        self.best_channels = best_channels or (lambda clusters, n=None: [])

        assert features
        self.features = features

        # This is a tuple (spikes, features, masks).
        self.background_features = background_features

        self.n_features_per_channel = n_features_per_channel
        assert n_channels > 0
        self.n_channels = n_channels

        # Spike times.
        self.n_spikes = spike_times.shape[0]
        assert spike_times.shape == (self.n_spikes,)
        assert self.n_spikes >= 0

        self.n_cols = self.n_features_per_channel + 1
        self.shape = (self.n_cols, self.n_cols)

        # Initialize the view.
        super(FeatureView, self).__init__(layout='grid',
                                          shape=self.shape,
                                          **kwargs)

        # Feature normalization.
        self.data_bounds = [-1, -feature_lim, +1, +feature_lim]

        # If this is True, the channels won't be automatically chosen
        # when new clusters are selected.
        self.fixed_channels = False

        # Channels to show.
        self.x_channels = None
        self.y_channels = None

        # Attributes: extra features. This is a dictionary
        # {name: (array, data_bounds)}
        # where each array is a `(n_spikes,)` array.
        self.attributes = {}
        self.add_attribute('time', spike_times)

    # Internal methods
    # -------------------------------------------------------------------------

    def _get_feature(self, dim, spike_ids, f):
        if dim in self.attributes:
            # Extra features like time.
            values, _ = self.attributes[dim]
            values = values[spike_ids]
            # assert values.shape == (f.shape[0],)
            return values
        else:
            assert len(dim) == 2
            ch, fet = dim
            return f[:, ch, fet] * self._scaling

    def _get_dim_bounds_single(self, dim):
        """Return the min and max of the bounds for a single dimension."""
        if dim in self.attributes:
            # Attribute: the data bounds were computed in add_attribute().
            y0, y1 = self.attributes[dim][1]
        else:
            # Features: the data bounds were computed in the constructor.
            _, y0, _, y1 = self.data_bounds
        return y0, y1

    def _get_dim_bounds(self, x_dim, y_dim):
        """Return the data bounds of a subplot, as a function of the
        two x-y dimensions."""
        x0, x1 = self._get_dim_bounds_single(x_dim)
        y0, y1 = self._get_dim_bounds_single(y_dim)
        return [x0, y0, x1, y1]

    def _plot_features(self, i, j, x_dim, y_dim, x, y,
                       masks=None, spike_clusters_rel=None):
        """Plot the features in a subplot."""
        assert x.shape == y.shape
        n_spikes = x.shape[0]

        sc = spike_clusters_rel
        if sc is not None:
            assert sc.shape == (n_spikes,)
        n_clusters = len(self.cluster_ids)

        # Retrieve the data bounds.
        data_bounds = self._get_dim_bounds(x_dim[i, j], y_dim[i, j])

        # Retrieve the masks and depth.
        mx, dx = _project_mask_depth(x_dim[i, j], masks,
                                     spike_clusters_rel=sc,
                                     n_clusters=n_clusters)
        my, dy = _project_mask_depth(y_dim[i, j], masks,
                                     spike_clusters_rel=sc,
                                     n_clusters=n_clusters)
        assert mx.shape == my.shape == dx.shape == dy.shape == (n_spikes,)

        d = np.maximum(dx, dy)
        m = np.maximum(mx, my)

        # Get the color of the markers.
        color = _get_color(m, spike_clusters_rel=sc, n_clusters=n_clusters)
        assert color.shape == (n_spikes, 4)

        # Create the scatter plot for the current subplot.
        # The marker size is smaller for background spikes.
        ms = (self._default_marker_size
              if spike_clusters_rel is not None else 1.)
        self[i, j].scatter(x=x,
                           y=y,
                           color=color,
                           depth=d,
                           data_bounds=data_bounds,
                           size=ms * np.ones(n_spikes),
                           )

    def _get_channel_dims(self, cluster_ids):
        """Select the channels to show by default."""
        n = self.n_cols - 1
        channels = self.best_channels(cluster_ids, 2 * n)
        channels = (channels if channels
                    else list(range(self.n_channels)))
        channels = _extend(channels, 2 * n)
        assert len(channels) == 2 * n
        return channels[:n], channels[n:]

    # Public methods
    # -------------------------------------------------------------------------

    def add_attribute(self, name, values, top_left=True):
        """Add an attribute (aka extra feature).

        The values should be a 1D array with `n_spikes` elements.

        By default, there is the `time` attribute.

        """
        assert values.shape == (self.n_spikes,)
        lim = values.min(), values.max()
        self.attributes[name] = (values, lim)
        # Register the attribute to use in the top-left subplot.
        if top_left:
            self.top_left_attribute = name

    def clear_channels(self):
        """Reset the dimensions."""
        self.x_channels = self.y_channels = None
        self.on_select()

    def on_select(self, cluster_ids=None):
        super(FeatureView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Get the spikes, features, masks.
        data = self.features(cluster_ids)
        spike_ids = data.spike_ids
        spike_clusters = data.spike_clusters
        f = data.features
        masks = data.masks
        assert f.ndim == 3
        assert masks.ndim == 2
        assert spike_ids.shape[0] == f.shape[0] == masks.shape[0]

        # Get the spike clusters.
        sc = _index_of(spike_clusters, cluster_ids)

        # Get the background features.
        data_bg = self.background_features
        spike_ids_bg = data_bg.spike_ids
        features_bg = data_bg.features
        masks_bg = data_bg.masks

        # Select the dimensions.
        # Choose the channels automatically unless fixed_channels is set.
        if (not self.fixed_channels or self.x_channels is None or
                self.y_channels is None):
            self.x_channels, self.y_channels = self._get_channel_dims(
                cluster_ids)
        tla = self.top_left_attribute
        assert self.x_channels
        assert self.y_channels
        x_dim, y_dim = _dimensions_matrix(self.x_channels, self.y_channels,
                                          n_cols=self.n_cols,
                                          top_left_attribute=tla)

        # Set the status message.
        ch_i = ', '.join(map(str, self.x_channels))
        ch_j = ', '.join(map(str, self.y_channels))
        self.set_status('Channels: {} - {}'.format(ch_i, ch_j))

        # Set a non-time attribute as y coordinate in the top-left subplot.
        attrs = sorted(self.attributes)
        attrs.remove('time')
        if attrs:
            y_dim[0, 0] = attrs[0]

        # Plot all features.
        with self.building():
            for i in range(self.n_cols):
                for j in range(self.n_cols):

                    # Retrieve the x and y values for the subplot.
                    x = self._get_feature(x_dim[i, j], spike_ids, f)
                    y = self._get_feature(y_dim[i, j], spike_ids, f)

                    # Retrieve the x and y values for the background spikes.
                    x_bg = self._get_feature(x_dim[i, j], spike_ids_bg,
                                             features_bg)
                    y_bg = self._get_feature(y_dim[i, j], spike_ids_bg,
                                             features_bg)

                    # Background features.
                    self._plot_features(i, j, x_dim, y_dim, x_bg, y_bg,
                                        masks=masks_bg)
                    # Cluster features.
                    self._plot_features(i, j, x_dim, y_dim, x, y,
                                        masks=masks,
                                        spike_clusters_rel=sc)

                    # Add axes.
                    self[i, j].lines(pos=[[-1, 0, +1, 0],
                                          [0, -1, 0, +1]],
                                     color=(.25, .25, .25, .5))

            # Add the boxes.
            self.grid.add_boxes(self, self.shape)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(FeatureView, self).attach(gui)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.add(self.clear_channels)
        self.actions.add(self.toggle_automatic_channel_selection)

        gui.connect_(self.on_channel_click)

    @property
    def state(self):
        return Bunch(scaling=self.scaling)

    def on_channel_click(self, channel_idx=None, key=None, button=None):
        """Respond to the click on a channel."""
        if key is None or not (1 <= key <= (self.n_cols - 1)):
            return
        # Get the axis from the pressed button (1, 2, etc.)
        axis = 'x' if button == 1 else 'y'
        # Get the existing channels.
        channels = self.x_channels if axis == 'x' else self.y_channels
        if channels is None:
            return
        assert len(channels) == self.n_cols - 1
        assert 0 <= channel_idx < self.n_channels
        # Update the channel.
        channels[key - 1] = channel_idx
        self.fixed_channels = True
        self.on_select()

    def toggle_automatic_channel_selection(self):
        """Toggle the automatic selection of channels when the cluster
        selection changes."""
        self.fixed_channels = not self.fixed_channels

    def increase(self):
        """Increase the scaling of the features."""
        self.scaling *= 1.2
        self.on_select()

    def decrease(self):
        """Decrease the scaling of the features."""
        self.scaling /= 1.2
        self.on_select()

    # Feature scaling
    # -------------------------------------------------------------------------

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class CorrelogramView(ManualClusteringView):
    excerpt_size = 10000
    n_excerpts = 100
    bin_size = 1e-3
    window_size = 50e-3
    uniform_normalization = False

    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
    }

    def __init__(self,
                 spike_times=None,
                 spike_clusters=None,
                 sample_rate=None,
                 **kwargs):

        assert sample_rate > 0
        self.sample_rate = sample_rate

        self.spike_times = np.asarray(spike_times)
        self.n_spikes, = self.spike_times.shape

        # Initialize the view.
        super(CorrelogramView, self).__init__(layout='grid',
                                              shape=(1, 1),
                                              **kwargs)

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters

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

    def _compute_correlograms(self, cluster_ids):

        # Keep spikes belonging to the selected clusters.
        ind = np.in1d(self.spike_clusters, cluster_ids)
        st = self.spike_times[ind]
        sc = self.spike_clusters[ind]

        # Take excerpts of the spikes.
        n_spikes_total = len(st)
        st = get_excerpts(st, excerpt_size=self.excerpt_size,
                          n_excerpts=self.n_excerpts)
        sc = get_excerpts(sc, excerpt_size=self.excerpt_size,
                          n_excerpts=self.n_excerpts)
        n_spikes_exerpts = len(st)
        logger.log(5, "Computing correlograms for clusters %s (%d/%d spikes).",
                   ', '.join(map(str, cluster_ids)),
                   n_spikes_exerpts, n_spikes_total,
                   )

        # Compute all pairwise correlograms.
        ccg = correlograms(st, sc,
                           cluster_ids=cluster_ids,
                           sample_rate=self.sample_rate,
                           bin_size=self.bin_size,
                           window_size=self.window_size,
                           )

        return ccg

    def on_select(self, cluster_ids=None):
        super(CorrelogramView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        ccg = self._compute_correlograms(cluster_ids)
        ylim = [ccg.max()] if not self.uniform_normalization else None

        colors = _selected_clusters_colors(n_clusters)

        self.grid.shape = (n_clusters, n_clusters)
        with self.building():
            for i in range(n_clusters):
                for j in range(n_clusters):
                    hist = ccg[i, j, :]
                    color = colors[i] if i == j else np.ones(3)
                    color = np.hstack((color, [1]))
                    self[i, j].hist(hist,
                                    color=color,
                                    ylim=ylim,
                                    )

    def toggle_normalization(self):
        """Change the normalization of the correlograms."""
        self.uniform_normalization = not self.uniform_normalization
        self.on_select()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(CorrelogramView, self).attach(gui)
        self.actions.add(self.toggle_normalization, shortcut='n')
        self.actions.add(self.set_bin, alias='cb')
        self.actions.add(self.set_window, alias='cw')

    @property
    def state(self):
        return Bunch(bin_size=self.bin_size,
                     window_size=self.window_size,
                     excerpt_size=self.excerpt_size,
                     n_excerpts=self.n_excerpts,
                     uniform_normalization=self.uniform_normalization,
                     )

    def set_bin(self, bin_size):
        """Set the correlogram bin size (in milliseconds)."""
        self.set_bin_window(bin_size=bin_size * 1e-3)
        self.on_select()

    def set_window(self, window_size):
        """Set the correlogram window size (in milliseconds)."""
        self.set_bin_window(window_size=window_size * 1e-3)
        self.on_select()


# -----------------------------------------------------------------------------
# Scatter view
# -----------------------------------------------------------------------------

class ScatterView(ManualClusteringView):
    _default_marker_size = 3.

    def __init__(self,
                 coords=None,  # function clusters: Bunch(x, y)
                 data_bounds=None,
                 **kwargs):

        assert coords
        self.coords = coords

        # Initialize the view.
        super(ScatterView, self).__init__(**kwargs)

        # Feature normalization.
        self.data_bounds = data_bounds

    def on_select(self, cluster_ids=None):
        super(ScatterView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Get the spike times and amplitudes
        data = self.coords(cluster_ids)
        spike_ids = data.spike_ids
        spike_clusters = data.spike_clusters
        x = data.x
        y = data.y
        n_spikes = len(spike_ids)
        assert n_spikes > 0
        assert spike_clusters.shape == (n_spikes,)
        assert x.shape == (n_spikes,)
        assert y.shape == (n_spikes,)

        # Get the spike clusters.
        sc = _index_of(spike_clusters, cluster_ids)

        # Plot the amplitudes.
        with self.building():
            m = np.ones(n_spikes)
            # Get the color of the markers.
            color = _get_color(m, spike_clusters_rel=sc, n_clusters=n_clusters)
            assert color.shape == (n_spikes, 4)
            ms = (self._default_marker_size if sc is not None else 1.)

            self.scatter(x=x,
                         y=y,
                         color=color,
                         data_bounds=self.data_bounds,
                         size=ms * np.ones(n_spikes),
                         )
