# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import inspect
from itertools import product
import logging
import re

import numpy as np
from vispy.util.event import Event

from phy.io.array import _index_of, _get_padded, get_excerpts
from phy.gui import Actions
from phy.plot import View, _get_linear_x
from phy.plot.utils import _get_boxes
from phy.stats import correlograms
from phy.utils import Bunch
from phy.utils._color import _spike_colors, ColorSelector, _colormap

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

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


def _extend(channels, n=None):
    channels = list(channels)
    if n is None:
        return channels
    if not len(channels):  # pragma: no cover
        channels = [0]
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
        'next_data': 'w',

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
                 best_channels=None,
                 **kwargs):
        self._key_pressed = None
        self._overlap = False
        self.do_zoom_on_channels = True
        self.do_show_labels = False
        self.data_index = 0

        self.best_channels = best_channels or (lambda clusters: [])

        # Channel positions and n_channels.
        assert channel_positions is not None
        self.channel_positions = np.asarray(channel_positions)
        self.n_channels = self.channel_positions.shape[0]

        # Initialize the view.
        box_bounds = _get_boxes(channel_positions, margin=.1)
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

        # Channel positions.
        assert channel_positions.shape == (self.n_channels, 2)
        self.channel_positions = channel_positions

    def on_select(self, cluster_ids=None):
        super(WaveformView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Load the waveform subset.
        data = self.waveforms(cluster_ids)
        # Take one element in the list.
        data = data[self.data_index % len(data)]
        alpha = data.get('alpha', .5)
        spike_ids = data.spike_ids
        spike_clusters = data.spike_clusters
        w = data.data
        masks = data.masks
        n_spikes = len(spike_ids)
        assert w.ndim == 3
        n_samples = w.shape[1]
        assert w.shape == (n_spikes, n_samples, self.n_channels)
        assert masks.shape == (n_spikes, self.n_channels)

        # Plot all waveforms.
        # OPTIM: avoid the loop.
        with self.building():
            already_shown = set()
            for i, cl in enumerate(cluster_ids):

                # Select the spikes corresponding to a given cluster.
                idx = spike_clusters == cl
                n_spikes_clu = idx.sum()  # number of spikes in the cluster.

                # Find the unmasked channels for those spikes.
                unmasked = np.nonzero(np.mean(masks[idx, :], axis=0) > .1)[0]
                n_unmasked = len(unmasked)
                assert n_unmasked > 0

                # Find the x coordinates.
                t = _get_linear_x(n_spikes_clu * n_unmasked, n_samples)
                if not self.overlap:
                    t = t + 2.5 * (i - (n_clusters - 1) / 2.)
                    # The total width should not depend on the number of
                    # clusters.
                    t /= n_clusters

                # Get the spike masks.
                m = masks[idx, :][:, unmasked].reshape((-1, 1))
                # HACK: on the GPU, we get the actual masks with fract(masks)
                # since we add the relative cluster index. We need to ensure
                # that the masks is never 1.0, otherwise it is interpreted as
                # 0.
                m *= .999
                # NOTE: we add the cluster index which is used for the
                # computation of the depth on the GPU.
                m += i

                color = tuple(_colormap(i)) + (alpha,)
                assert len(color) == 4

                # Generate the box index (one number per channel).
                box_index = unmasked
                box_index = np.repeat(box_index, n_samples)
                box_index = np.tile(box_index, n_spikes_clu)
                assert box_index.shape == (n_spikes_clu * n_unmasked *
                                           n_samples)

                # Generate the waveform array.
                wave = w[idx, :, :][:, :, unmasked]
                wave = np.transpose(wave, (0, 2, 1))
                wave = wave.reshape((n_spikes_clu * n_unmasked, n_samples))

                self.plot(x=t,
                          y=wave,
                          color=color,
                          masks=m,
                          box_index=box_index,
                          data_bounds=None,
                          uniform=True,
                          )
                # Add channel labels.
                if self.do_show_labels:
                    for ch in unmasked:
                        # Skip labels that have already been shown.
                        if ch in already_shown:
                            continue
                        already_shown.add(ch)
                        self[ch].text(pos=[t[0, 0], 0.],
                                      # TODO: use real channel labels.
                                      text=str(ch),
                                      anchor=[-1.01, -.25],
                                      data_bounds=None,
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
                     do_show_labels=self.do_show_labels,
                     )

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(WaveformView, self).attach(gui)
        self.actions.add(self.toggle_waveform_overlap)
        self.actions.add(self.toggle_zoom_on_channels)
        self.actions.add(self.toggle_show_labels)

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

        self.actions.add(self.next_data)

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

    def next_data(self):
        """Show the next set of waveforms (if any)."""
        # HACK: temporarily disable automatic zoom on channels when
        # changing the data.
        tmp = self.do_zoom_on_channels
        self.do_zoom_on_channels = False
        self.data_index += 1
        self.on_select()
        self.do_zoom_on_channels = tmp

    def toggle_zoom_on_channels(self):
        self.do_zoom_on_channels = not self.do_zoom_on_channels

    def toggle_show_labels(self):
        self.do_show_labels = not self.do_show_labels
        tmp = self.do_zoom_on_channels
        self.do_zoom_on_channels = False
        self.on_select()
        self.do_zoom_on_channels = tmp

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
        # Center of the new range.
        cx = (x0 + x1) * .5
        cy = (y0 + y1) * .5
        # Previous range.
        px0, py0, px1, py1 = self.panzoom.get_range()
        # Half-size of the previous range.
        dx = (px1 - px0) * .5
        dy = (py1 - py0) * .5
        # New range.
        new_range = (cx - dx, cy - dy, cx + dx, cy + dy)
        self.panzoom.set_range(new_range)

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
    traces = traces - np.mean(traces, axis=0)
    return traces


def extract_spikes(traces, interval, sample_rate=None,
                   spike_times=None, spike_clusters=None,
                   cluster_groups=None,
                   all_masks=None,
                   n_samples_waveforms=None):
    cluster_groups = cluster_groups or {}
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
        o = _extract_wave(traces, sample_start, m[i], wave_len)
        if o is None:  # pragma: no cover
            logger.debug("Unable to extract spike %d.", i)
            continue
        b.waveforms, b.channels = o
        # Masks on unmasked channels.
        b.masks = m[i, b.channels]
        b.spike_time = st[i]
        b.spike_cluster = sc[i]
        b.cluster_group = cluster_groups.get(b.spike_cluster, None)
        b.offset_samples = offset_samples

        spikes.append(b)
    return spikes


class TraceView(ManualClusteringView):
    interval_duration = .25  # default duration of the interval
    shift_amount = .1
    scaling_coeff_x = 1.5
    scaling_coeff_y = 1.1
    default_trace_color = (.75, .75, .75, .75)
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

        self.do_show_labels = False

        # traces is a function interval => [traces]
        # spikes is a function interval => [Bunch(...)]

        # Sample rate.
        assert sample_rate > 0
        self.sample_rate = float(sample_rate)
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

        self._color_selector = ColorSelector()

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
        self._interval = None
        self.go_to(duration / 2.)

    # Internal methods
    # -------------------------------------------------------------------------

    def _plot_traces(self, traces=None, color=None):
        traces = traces.T
        n_samples = traces.shape[1]
        n_ch = self.n_channels
        assert traces.shape == (n_ch, n_samples)
        color = color or self.default_trace_color

        t = self._interval[0] + np.arange(n_samples) * self.dt
        t = self._normalize_time(t)
        t = np.tile(t, (n_ch, 1))
        box_index = np.repeat(np.arange(n_ch)[:, np.newaxis],
                              n_samples,
                              axis=1)

        assert t.shape == (n_ch, n_samples)
        assert traces.shape == (n_ch, n_samples)
        assert box_index.shape == (n_ch, n_samples)

        self.plot(t, traces,
                  color=color,
                  data_bounds=None,
                  box_index=box_index,
                  uniform=True,
                  )

    def _plot_spike(self, waveforms=None, channels=None, spike_time=None,
                    offset_samples=0, color=None):

        n_samples, n_channels = waveforms.shape
        assert len(channels) == n_channels
        sr = float(self.sample_rate)

        t0 = spike_time - offset_samples / sr

        # Generate the x coordinates of the waveform.
        t = t0 + self.dt * np.arange(n_samples)
        t = self._normalize_time(t)
        t = np.tile(t, (n_channels, 1))  # (n_unmasked_channels, n_samples)

        # The box index depends on the channel.
        box_index = np.repeat(channels[:, np.newaxis], n_samples, axis=0)
        self.plot(t, waveforms.T, color=color,
                  box_index=box_index,
                  data_bounds=None,
                  )

    def _plot_labels(self, traces):
        for ch in range(self.n_channels):
            self[ch].text(pos=[-1., traces[0, ch]],
                          text=str(ch),
                          anchor=[+1., -.1],
                          data_bounds=None,
                          )

    def _restrict_interval(self, interval):
        start, end = interval
        # Round the times to full samples to avoid subsampling shifts
        # in the traces.
        start = int(round(start * self.sample_rate)) / self.sample_rate
        end = int(round(end * self.sample_rate)) / self.sample_rate
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
        if interval == self._interval:
            return
        self.clear()
        interval = self._restrict_interval(interval)
        self._interval = interval
        start, end = interval

        # OPTIM: normalize time manually into [-1.0, 1.0].
        def _normalize_time(t):
            return -1. + (2. / float(end - start)) * (t - start)
        self._normalize_time = _normalize_time

        # Set the status message.
        if change_status:
            self.set_status('Interval: {:.3f} s - {:.3f} s'.format(start, end))

        # Load the traces.
        all_traces = self.traces(interval)
        assert isinstance(all_traces, (tuple, list))

        # Plot the traces.
        for i, traces in enumerate(all_traces):
            # Only show labels for the first set of traces.
            self._plot_traces(**traces)

        # Plot the spikes.
        spikes = self.spikes(interval, all_traces)
        assert isinstance(spikes, (tuple, list))
        for spike in spikes:
            clu = spike.spike_cluster
            cg = spike.cluster_group
            color = self._color_selector.get(clu,
                                             cluster_ids=self.cluster_ids,
                                             cluster_group=cg,
                                             )
            self._plot_spike(color=color,
                             waveforms=spike.waveforms,
                             channels=spike.channels,
                             spike_time=spike.spike_time,
                             offset_samples=spike.offset_samples,
                             )

        # Plot the labels.
        if self.do_show_labels:
            self._plot_labels(all_traces[0].traces)

        self.build()
        self.update()

    def on_select(self, cluster_ids=None):
        super(TraceView, self).on_select(cluster_ids)
        self.set_interval(self._interval, change_status=False)

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
        self.actions.add(self.toggle_show_labels)

    @property
    def state(self):
        return Bunch(scaling=self.scaling,
                     origin=self.origin,
                     interval=self._interval,
                     do_show_labels=self.do_show_labels,
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
        return sum(self._interval) * .5

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        self.set_interval(value)

    @property
    def half_duration(self):
        """Half of the duration of the current interval."""
        if self._interval is not None:
            a, b = self._interval
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
        start, end = self._interval
        delay = (end - start) * .2
        self.shift(delay)

    def go_left(self):
        """Go to left."""
        start, end = self._interval
        delay = (end - start) * .2
        self.shift(-delay)

    def widen(self):
        """Increase the interval size."""
        t, h = self.time, self.half_duration
        h *= self.scaling_coeff_x
        self.set_interval((t - h, t + h))

    def narrow(self):
        """Decrease the interval size."""
        t, h = self.time, self.half_duration
        h /= self.scaling_coeff_x
        self.set_interval((t - h, t + h))

    def toggle_show_labels(self):
        self.do_show_labels = not self.do_show_labels
        self.on_select()

    # Channel scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.stacked.box_size = self.box_size * self.scaling

    def increase(self):
        """Increase the scaling of the traces."""
        self.scaling *= self.scaling_coeff_y
        self._update_boxes()

    def decrease(self):
        """Decrease the scaling of the traces."""
        self.scaling /= self.scaling_coeff_y
        self._update_boxes()


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------

def _dimensions_matrix(channels, n_cols=None, top_left_attribute=None):
    """
    time,x0 y0,x0   x1,x0   y1,x0
    x0,y0   time,y0 x1,y0   y1,y0
    x0,x1   y0,x1   time,x1 y1,x1
    x0,y1   y0,y1   x1,y1   time,y1
    """
    # Generate the dimensions matrix from the docstring.
    ds = inspect.getdoc(_dimensions_matrix).strip()
    x, y = channels[:2]

    def _get_dim(d):
        if d == 'time':
            return d
        assert re.match(r'[xy][01]', d)
        c = x if d[0] == 'x' else y
        f = int(d[1])
        return c, f

    dims = [[_.split(',') for _ in re.split(r' +', line.strip())]
            for line in ds.splitlines()]
    x_dim = {(i, j): _get_dim(dims[i][j][0])
             for i, j in product(range(4), range(4))}
    y_dim = {(i, j): _get_dim(dims[i][j][1])
             for i, j in product(range(4), range(4))}
    return x_dim, y_dim


class FeatureView(ManualClusteringView):
    _default_marker_size = 3.

    default_shortcuts = {
        'increase': 'ctrl++',
        'decrease': 'ctrl+-',
        'toggle_automatic_channel_selection': 'c',
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

        self.best_channels = best_channels or (lambda clusters=None: [])

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
        self.spike_times = spike_times
        self.duration = spike_times.max()

        self.n_cols = 4
        self.shape = (self.n_cols, self.n_cols)

        # Initialize the view.
        super(FeatureView, self).__init__(layout='grid',
                                          shape=self.shape,
                                          enable_lasso=True,
                                          **kwargs)

        # If this is True, the channels won't be automatically chosen
        # when new clusters are selected.
        self.fixed_channels = False

        # Channels to show.
        self.channels = None

        # Attributes: extra features. This is a dictionary
        # {name: array}
        # where each array is a `(n_spikes,)` array.
        self.attributes = {}
        self.top_left_attribute = None

    # Internal methods
    # -------------------------------------------------------------------------

    def _get_feature(self, dim, spike_ids, f):
        if dim == 'time':
            return -1. + (2. / self.duration) * self.spike_times[spike_ids]
        elif dim in self.attributes:
            # Extra features.
            values = self.attributes[dim]
            values = values[spike_ids]
            return values
        else:
            assert len(dim) == 2
            ch, fet = dim
            assert fet < f.shape[2]
            return f[:, ch, fet] * self._scaling

    def _plot_features(self, i, j, x_dim, y_dim, x, y,
                       masks=None, clu_idx=None):
        """Plot the features in a subplot."""
        assert x.shape == y.shape
        n_spikes = x.shape[0]

        if clu_idx is not None:
            color = tuple(_colormap(clu_idx)) + (.5,)
        else:
            color = (1., 1., 1., .5)
        assert len(color) == 4

        # Find the masks for the given subplot channel.
        if isinstance(x_dim[i, j], tuple):
            ch, fet = x_dim[i, j]
            # NOTE: we add the cluster relative index for the computation
            # of the depth on the GPU.
            m = masks[:, ch] * .999 + (clu_idx or 0)
        else:
            m = np.ones(n_spikes) * .999 + (clu_idx or 0)

        # Marker size, smaller for background features.
        size = self._default_marker_size if clu_idx is not None else 1.

        self[i, j].scatter(x=x, y=y,
                           color=color,
                           masks=m,
                           size=size,
                           data_bounds=None,
                           uniform=True,
                           )
        if i == 0:
            # HACK: call this when i=0 (first line) but plot the text
            # in the last subplot line. This is because we skip i > j
            # in the subplot loop.
            i0 = (self.n_cols - 1)
            dim = x_dim[i0, j] if j < (self.n_cols - 1) else x_dim[i0, 0]
            self[i0, j].text(pos=[0., -1.],
                             text=str(dim),
                             anchor=[0., -1.04],
                             data_bounds=None,
                             )
        if j == 0:
            self[i, j].text(pos=[-1., 0.],
                            text=str(y_dim[i, j]),
                            anchor=[-1.03, 0.],
                            data_bounds=None,
                            )

    def _get_channel_dims(self, cluster_ids):
        """Select the channels to show by default."""
        n = 2
        channels = self.best_channels(cluster_ids)
        channels = (channels if channels is not None
                    else list(range(self.n_channels)))
        channels = _extend(channels, n)
        assert len(channels) == n
        return channels

    # Public methods
    # -------------------------------------------------------------------------

    def add_attribute(self, name, values, top_left=True):
        """Add an attribute (aka extra feature).

        The values should be a 1D array with `n_spikes` elements.

        NOTE: the values should be normalized by the caller.

        """
        assert values.shape == (self.n_spikes,)
        self.attributes[name] = values
        # Register the attribute to use in the top-left subplot.
        if top_left:
            self.top_left_attribute = name

    def clear_channels(self):
        """Reset the dimensions."""
        self.channels = None
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
        f = data.data
        masks = data.masks
        assert f.ndim == 3
        assert masks.ndim == 2
        assert spike_ids.shape[0] == f.shape[0] == masks.shape[0]

        # Get the background features.
        data_bg = self.background_features
        if data_bg is not None:
            spike_ids_bg = data_bg.spike_ids
            features_bg = data_bg.data
            masks_bg = data_bg.masks

        # Select the dimensions.
        # Choose the channels automatically unless fixed_channels is set.
        if (not self.fixed_channels or self.channels is None):
            self.channels = self._get_channel_dims(cluster_ids)
        tla = self.top_left_attribute
        assert self.channels
        x_dim, y_dim = _dimensions_matrix(self.channels,
                                          n_cols=self.n_cols,
                                          top_left_attribute=tla)

        # Set the status message.
        ch = ', '.join(map(str, self.channels))
        self.set_status('Channels: {}'.format(ch))

        # Set a non-time attribute as y coordinate in the top-left subplot.
        attrs = sorted(self.attributes)
        # attrs.remove('time')
        if attrs:
            y_dim[0, 0] = attrs[0]

        # Plot all features.
        with self.building():
            for i in range(self.n_cols):
                for j in range(self.n_cols):
                    # Skip lower-diagonal subplots.
                    if i > j:
                        continue

                    # Retrieve the x and y values for the subplot.
                    x = self._get_feature(x_dim[i, j], spike_ids, f)
                    y = self._get_feature(y_dim[i, j], spike_ids, f)

                    if data_bg is not None:
                        # Retrieve the x and y values for the background
                        # spikes.
                        x_bg = self._get_feature(x_dim[i, j], spike_ids_bg,
                                                 features_bg)
                        y_bg = self._get_feature(y_dim[i, j], spike_ids_bg,
                                                 features_bg)

                        # Background features.
                        self._plot_features(i, j, x_dim, y_dim, x_bg, y_bg,
                                            masks=masks_bg,
                                            )

                    # Cluster features.
                    for clu_idx, clu in enumerate(cluster_ids):
                        # TODO: compute this only once, outside the loop.
                        idx = data.spike_clusters == clu
                        self._plot_features(i, j, x_dim, y_dim,
                                            x[idx], y[idx],
                                            masks=masks[idx],
                                            clu_idx=clu_idx,
                                            )

                    # Add axes.
                    self[i, j].lines(pos=[[-1., 0., +1., 0.],
                                          [0., -1., 0., +1.]],
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
        gui.connect_(self.on_request_split)

    @property
    def state(self):
        return Bunch(scaling=self.scaling)

    def on_channel_click(self, channel_idx=None, key=None, button=None):
        """Respond to the click on a channel."""
        channels = self.channels
        if channels is None:
            return
        assert len(channels) == 2
        assert 0 <= channel_idx < self.n_channels
        # Get the axis from the pressed button (1, 2, etc.)
        # axis = 'x' if button == 1 else 'y'
        channels[0 if button == 1 else 1] = channel_idx
        self.fixed_channels = True
        self.on_select()

    def on_request_split(self):
        """Return the spikes enclosed by the lasso."""
        if self.lasso.count < 3:
            return []
        tla = self.top_left_attribute
        assert self.channels
        x_dim, y_dim = _dimensions_matrix(self.channels,
                                          n_cols=self.n_cols,
                                          top_left_attribute=tla)
        data = self.features(self.cluster_ids, load_all=True)
        spike_ids = data.spike_ids
        f = data.data
        i, j = self.lasso.box

        x = self._get_feature(x_dim[i, j], spike_ids, f)
        y = self._get_feature(y_dim[i, j], spike_ids, f)
        pos = np.c_[x, y].astype(np.float64)

        ind = self.lasso.in_polygon(pos)
        self.lasso.clear()
        return spike_ids[ind]

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
        self.sample_rate = float(sample_rate)

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

        colors = _spike_colors(np.arange(n_clusters), alpha=1.)

        self.grid.shape = (n_clusters, n_clusters)
        with self.building():
            for i in range(n_clusters):
                for j in range(n_clusters):
                    hist = ccg[i, j, :]
                    color = colors[i] if i == j else np.ones(4)
                    self[i, j].hist(hist,
                                    color=color,
                                    ylim=ylim,
                                    )
                    # Cluster labels.
                    if i == (n_clusters - 1):
                        self[i, j].text(pos=[0., -1.],
                                        text=str(cluster_ids[j]),
                                        anchor=[0., -1.04],
                                        data_bounds=None,
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
                 **kwargs):

        assert coords
        self.coords = coords

        # Initialize the view.
        super(ScatterView, self).__init__(**kwargs)

    def on_select(self, cluster_ids=None):
        super(ScatterView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Get the spike times and amplitudes
        data = self.coords(cluster_ids)
        if data is None:
            self.clear()
            return
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
            color = _spike_colors(sc, masks=m)
            assert color.shape == (n_spikes, 4)
            ms = (self._default_marker_size if sc is not None else 1.)

            self.scatter(x=x,
                         y=y,
                         color=color,
                         size=ms * np.ones(n_spikes),
                         )
