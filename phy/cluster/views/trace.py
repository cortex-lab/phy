# -*- coding: utf-8 -*-

"""Trace view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils import Bunch
from phy.utils._color import ColorSelector
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Trace view
# -----------------------------------------------------------------------------

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
        'toggle_show_labels': 'alt+l',
        'widen': 'alt+-',
        'narrow': 'alt++',
    }

    def __init__(self,
                 traces=None,
                 spikes=None,
                 sample_rate=None,
                 duration=None,
                 n_channels=None,
                 channel_positions=None,
                 channel_order=None,
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

        channel_positions = (channel_positions
                             if channel_positions is not None
                             else np.c_[np.arange(n_channels),
                                        np.zeros(n_channels)])
        assert channel_positions.shape == (n_channels, 2)
        self.channel_positions = channel_positions

        channel_order = (channel_order if channel_order is not None
                         else np.arange(n_channels))
        assert channel_order.shape == (n_channels,)
        self.channel_order = channel_order

        # Double argsort for inverse permutation.
        self.channel_vertical_order = \
            np.argsort(np.argsort(channel_positions[:, 1]))

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
        # Display the channels in vertical order.
        order = self.channel_vertical_order
        box_index = np.repeat(order[:, np.newaxis],
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
        c = self.channel_vertical_order[channels]
        box_index = np.repeat(c[:, np.newaxis], n_samples, axis=0)
        self.plot(t, waveforms.T, color=color,
                  box_index=box_index,
                  data_bounds=None,
                  )

    def _plot_labels(self, traces):
        for ch in range(self.n_channels):
            ch_label = '%d' % self.channel_order[ch]
            och = self.channel_vertical_order[ch]
            self[och].text(pos=[-1., traces[0, ch]],
                           text=ch_label,
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
