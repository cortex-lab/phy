# -*- coding: utf-8 -*-

"""Trace view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from phy.plot.transform import NDC, Range
from vispy.util.event import Event

from phy.utils import Bunch
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Trace view
# -----------------------------------------------------------------------------

def select_traces(traces, interval, sample_rate=None):
    """Load traces in an interval (in seconds)."""
    start, end = interval
    i, j = round(sample_rate * start), round(sample_rate * end)
    i, j = int(i), int(j)
    traces = traces[i:j, :]
    # traces = traces - np.mean(traces, axis=0)
    return traces


def _iter_spike_waveforms(interval=None,
                          traces_interval=None,
                          model=None,
                          supervisor=None,
                          n_samples_waveforms=None,
                          get_best_channels=None,
                          show_all_spikes=False,
                          color_selector=None,
                          ):
    m = model
    p = supervisor
    cs = color_selector
    sr = m.sample_rate
    a, b = m.spike_times.searchsorted(interval)
    s0, s1 = int(round(interval[0] * sr)), int(round(interval[1] * sr))
    ns = n_samples_waveforms
    k = ns // 2
    for i in range(a, b):
        t = m.spike_times[i]
        c = m.spike_clusters[i]
        # Skip non-selected spikes if requested.
        if (not show_all_spikes and c not in supervisor.selected):
            continue
        cg = p.cluster_meta.get('group', c)
        channel_ids = get_best_channels(c)
        s = int(round(t * sr)) - s0
        # Skip partial spikes.
        if s - k < 0 or s + k >= (s1 - s0):  # pragma: no cover
            continue
        color = cs.get(c, cluster_ids=p.selected, cluster_group=cg)
        # Extract the waveform.
        wave = Bunch(data=traces_interval[s - k:s + ns - k, channel_ids],
                     channel_ids=channel_ids,
                     start_time=(s + s0 - k) / sr,
                     color=color,
                     spike_id=i,
                     spike_time=t,
                     spike_cluster=c,
                     cluster_group=cg,
                     )
        assert wave.data.shape[0] == ns
        yield wave


class SpikeClick(Event):
    def __init__(self, type, channel_id=None, spike_id=None, cluster_id=None):
        super(SpikeClick, self).__init__(type)
        self.channel_id = channel_id
        self.spike_id = spike_id
        self.cluster_id = cluster_id


class TraceView(ManualClusteringView):
    interval_duration = .25  # default duration of the interval
    shift_amount = .1
    scaling_coeff_x = 1.5
    scaling_coeff_y = 1.1
    default_trace_color = (.75, .75, .75, 1.)
    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
        'decrease': 'alt+down',
        'increase': 'alt+up',
        'toggle_show_labels': 'alt+l',
        'widen': 'alt+-',
        'narrow': 'alt++',
    }

    def __init__(self,
                 traces=None,
                 sample_rate=None,
                 duration=None,
                 n_channels=None,
                 channel_vertical_order=None,
                 **kwargs):

        self.do_show_labels = None
        self._key_pressed = None

        # traces is a function interval => [traces]
        # spikes is a function interval => [Bunch(...)]

        # Sample rate.
        assert sample_rate > 0
        self.sample_rate = float(sample_rate)
        self.dt = 1. / self.sample_rate

        # Traces and spikes.
        assert hasattr(traces, '__call__')
        self.traces = traces

        assert duration >= 0
        self.duration = duration

        assert n_channels >= 0
        self.n_channels = n_channels

        assert (channel_vertical_order is None or
                channel_vertical_order.shape == (n_channels,))
        self._channel_perm = channel_vertical_order
        if self._channel_perm is not None:
            self._channel_perm = np.argsort(self._channel_perm)

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
        self._interval = None
        self.go_to(duration / 2.)

        self._waveform_times = []
        self.events.add(spike_click=SpikeClick)

    def _permute_channels(self, x, inv=False):
        cp = self._channel_perm
        cp = np.argsort(cp) if inv and cp is not None else cp
        return cp[x] if cp is not None else x

    # Internal methods
    # -------------------------------------------------------------------------

    def _plot_traces(self, traces, color=None, data_bounds=None):
        traces = traces.T
        n_samples = traces.shape[1]
        n_ch = self.n_channels
        assert traces.shape == (n_ch, n_samples)
        color = color or self.default_trace_color

        t = self._interval[0] + np.arange(n_samples) * self.dt
        t = np.tile(t, (n_ch, 1))

        box_index = self._permute_channels(np.arange(n_ch))
        box_index = np.repeat(box_index[:, np.newaxis],
                              n_samples,
                              axis=1)

        assert t.shape == (n_ch, n_samples)
        assert traces.shape == (n_ch, n_samples)
        assert box_index.shape == (n_ch, n_samples)

        self.uplot(t, traces,
                   color=color,
                   data_bounds=data_bounds,
                   box_index=box_index,
                   )

    def _plot_waveforms(self, waveforms=None,
                        channel_ids=None,
                        start_time=None,
                        color=None,
                        data_bounds=None,
                        ):
        # The spike time corresponds to the first sample of the waveform.
        n_samples, n_channels = waveforms.shape

        # Generate the x coordinates of the waveform.
        t = start_time + self.dt * np.arange(n_samples)
        t = np.tile(t, (n_channels, 1))  # (n_unmasked_channels, n_samples)

        # The box index depends on the channel.
        box_index = self._permute_channels(channel_ids)
        box_index = np.repeat(box_index[:, np.newaxis], n_samples, axis=0)
        self.plot(t, waveforms.T, color=color,
                  box_index=box_index,
                  data_bounds=data_bounds,
                  )

    def _plot_labels(self, traces, data_bounds=None):
        for ch in range(self.n_channels):
            bi = self._permute_channels(ch)
            ch_label = '%d' % ch
            self[bi].text(pos=[data_bounds[0], traces[0, ch]],
                          text=ch_label,
                          anchor=[+1., -.1],
                          data_bounds=data_bounds,
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

    def set_interval(self, interval=None, change_status=True,
                     force_update=False):
        """Display the traces and spikes in a given interval."""
        if interval is None:
            interval = self._interval
        interval = self._restrict_interval(interval)
        if not force_update and interval == self._interval:
            return
        self._interval = interval
        start, end = interval
        self.clear()

        # Set the status message.
        if change_status:
            self.set_status('Interval: {:.3f} s - {:.3f} s'.format(start, end))

        # Load the traces.
        traces = self.traces(interval)

        # Find the data bounds.
        ymin, ymax = traces.data.min(), traces.data.max()
        data_bounds = (start, ymin, end, ymax)

        # Used for spike click.
        self._data_bounds = data_bounds
        self._waveform_times = []

        # Plot the traces.
        self._plot_traces(traces.data,
                          color=traces.get('color', None),
                          data_bounds=data_bounds,
                          )

        # Plot the spikes.
        waveforms = traces.waveforms
        assert isinstance(waveforms, list)
        for w in waveforms:
            self._plot_waveforms(waveforms=w.data,
                                 color=w.color,
                                 channel_ids=w.get('channel_ids', None),
                                 start_time=w.start_time,
                                 data_bounds=data_bounds,
                                 )
            self._waveform_times.append((w.start_time,
                                         w.spike_id,
                                         w.spike_cluster,
                                         w.get('channel_ids', None),
                                         ))

        # Plot the labels.
        if self.do_show_labels:
            self._plot_labels(traces.data, data_bounds=data_bounds)

        self.build()
        self.update()

    def on_select(self, cluster_ids=None, **kwargs):
        super(TraceView, self).on_select(cluster_ids, **kwargs)
        self.set_interval(self._interval, change_status=False,
                          force_update=kwargs.get('force_update', None))

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(TraceView, self).attach(gui)
        self.actions.add(self.go_to, alias='tg')
        self.actions.separator()
        self.actions.add(self.shift, alias='ts')
        self.actions.add(self.go_right)
        self.actions.add(self.go_left)
        self.actions.separator()
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.separator()
        self.actions.add(self.widen)
        self.actions.add(self.narrow)
        self.actions.separator()
        self.actions.add(self.toggle_show_labels)

        # We forward the event from VisPy to the phy GUI.
        @self.connect
        def on_spike_click(e):
            logger.log(5, "Spike click on channel %s, spike %s, cluster %s.",
                       e.channel_id, e.spike_id, e.cluster_id)
            gui.emit('spike_click',
                     channel_id=e.channel_id,
                     spike_id=e.spike_id,
                     cluster_id=e.cluster_id,
                     )

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
        self.set_interval(force_update=True)

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

    # Spike selection
    # -------------------------------------------------------------------------

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
            box_id = self.stacked.get_closest_box(mouse_pos)
            channel_id = self._permute_channels(box_id, inv=True)
            # Find the spike and cluster closest to the mouse.
            db = self._data_bounds
            # Get the information about the displayed spikes.
            if not self._waveform_times:
                return
            # Get the time coordinate of the mouse position.
            mouse_time = Range(NDC, db).apply(mouse_pos)[0][0]
            # Get the closest spike id.
            times, spike_ids, spike_clusters, channel_ids = \
                zip(*(_ for _ in self._waveform_times if channel_id in _[3]))
            i = np.argmin(np.abs(np.array(times) - mouse_time))
            # Raise the spike_click event.
            spike_id = spike_ids[i]
            cluster_id = spike_clusters[i]
            self.events.spike_click(channel_id=channel_id,
                                    spike_id=spike_id,
                                    cluster_id=cluster_id
                                    )

    def on_key_release(self, event):
        self._key_pressed = None
