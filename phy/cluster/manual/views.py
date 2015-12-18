# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from phy.io.array import _index_of, _get_padded, get_excerpts
from phy.gui import Actions
from phy.plot import View, _get_linear_x
from phy.plot.utils import _get_boxes
from phy.stats import correlograms
from phy.stats.clusters import mean, unmasked_channels, sorted_main_channels
from phy.utils import IPlugin

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


def _extract_wave(traces, spk, mask, wave_len=None):
    n_samples, n_channels = traces.shape
    if not (0 <= spk < n_samples):
        raise ValueError()
    assert mask.shape == (n_channels,)
    channels = np.nonzero(mask > .1)[0]
    # There should be at least one non-masked channel.
    if not len(channels):
        return  # pragma: no cover
    i = spk - wave_len // 2
    j = spk + wave_len // 2
    a, b = max(0, i), min(j, n_samples - 1)
    data = traces[a:b, channels]
    data = _get_padded(data, i - a, i - a + wave_len)
    assert data.shape == (wave_len, len(channels))
    return data, channels


def _get_data_bounds(arr, n_spikes=None, percentile=None):
    n = arr.shape[0]
    k = max(1, n // n_spikes) if n_spikes else 1
    w = np.abs(arr[::k])
    n = w.shape[0]
    w = w.reshape((n, -1))
    w = w.max(axis=1)
    m = np.percentile(w, percentile)
    return [-1, -m, +1, +m]


def _get_spike_clusters_rel(spike_clusters, spike_ids, cluster_ids):
    # Relative spike clusters.
    # NOTE: the order of the clusters in cluster_ids matters.
    # It will influence the relative index of the clusters, which
    # in return influence the depth.
    spike_clusters = spike_clusters[spike_ids]
    assert np.all(np.in1d(spike_clusters, cluster_ids))
    spike_clusters_rel = _index_of(spike_clusters, cluster_ids)
    return spike_clusters_rel


def _get_depth(masks, spike_clusters_rel=None, n_clusters=None):
    """Return the OpenGL z-depth of vertices as a function of the
    mask and cluster index."""
    n_spikes = len(masks)
    assert masks.shape == (n_spikes,)
    depth = (-0.1 - (spike_clusters_rel + masks) /
             float(n_clusters + 10.))
    depth[masks <= 0.25] = 0
    assert depth.shape == (n_spikes,)
    return depth


def _get_color(masks, spike_clusters_rel=None, n_clusters=None):
    """Return the color of vertices as a function of the mask and
    cluster index."""
    n_spikes = len(masks)
    assert masks.shape == (n_spikes,)
    assert spike_clusters_rel.shape == (n_spikes,)
    # Generate the colors.
    colors = _selected_clusters_colors(n_clusters)
    # Color as a function of the mask.
    color = colors[spike_clusters_rel]
    hsv = rgb_to_hsv(color[:, :3])
    # Change the saturation and value as a function of the mask.
    hsv[:, 1] *= masks
    hsv[:, 2] *= .5 * (1. + masks)
    color = hsv_to_rgb(hsv)
    color = np.c_[color, .5 * np.ones((n_spikes, 1))]
    return color


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

class ManualClusteringView(View):
    default_shortcuts = {
    }

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        self.cluster_ids = None
        self.spike_ids = None

        super(ManualClusteringView, self).__init__(**kwargs)

    def on_select(self, cluster_ids=None, spike_ids=None):
        cluster_ids = (cluster_ids if cluster_ids is not None
                       else self.cluster_ids)
        spike_ids = (spike_ids if spike_ids is not None
                     else self.spike_ids)
        self.cluster_ids = list(cluster_ids)
        self.spike_ids = spike_ids

    def attach(self, gui):
        """Attach the view to the GUI."""

        # Disable keyboard pan so that we can use arrows as global shortcuts
        # in the GUI.
        self.panzoom.enable_keyboard_pan = False

        gui.add_view(self)
        gui.connect_(self.on_select)
        self.actions = Actions(gui, default_shortcuts=self.shortcuts)

        self.show()


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

class WaveformView(ManualClusteringView):
    normalization_percentile = .95
    normalization_n_spikes = 1000
    overlap = False
    box_coeff = 1.1

    default_shortcuts = {
        'toggle_waveform_overlap': 'o',

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
                 masks=None,
                 spike_clusters=None,
                 channel_positions=None,
                 box_scaling=None,
                 probe_scaling=None,
                 **kwargs):
        """

        The channel order in waveforms needs to correspond to the one
        in channel_positions.

        """
        # Initialize the view.
        box_bounds = _get_boxes(channel_positions)
        super(WaveformView, self).__init__(layout='boxed',
                                           box_bounds=box_bounds,
                                           **kwargs)

        # Box and probe scaling.
        self.box_scaling = np.array(box_scaling if box_scaling is not None
                                    else (1., 1.))
        self.probe_scaling = np.array(probe_scaling
                                      if probe_scaling is not None
                                      else (1., 1.))

        # Make a copy of the initial box pos and size. We'll apply the scaling
        # to these quantities.
        self.box_pos = np.array(self.boxed.box_pos)
        self.box_size = np.array(self.boxed.box_size)
        self._update_boxes()

        # Waveforms.
        assert waveforms.ndim == 3
        self.n_spikes, self.n_samples, self.n_channels = waveforms.shape
        self.waveforms = waveforms

        # Waveform normalization.
        self.data_bounds = _get_data_bounds(waveforms,
                                            self.normalization_n_spikes,
                                            self.normalization_percentile)

        # Masks.
        self.masks = masks

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters

        # Channel positions.
        assert channel_positions.shape == (self.n_channels, 2)
        self.channel_positions = channel_positions

    def on_select(self, cluster_ids=None, spike_ids=None):
        super(WaveformView, self).on_select(cluster_ids=cluster_ids,
                                            spike_ids=spike_ids)
        cluster_ids, spike_ids = self.cluster_ids, self.spike_ids
        n_clusters = len(cluster_ids)
        n_spikes = len(spike_ids)
        if n_spikes == 0:
            return

        # Relative spike clusters.
        spike_clusters_rel = _get_spike_clusters_rel(self.spike_clusters,
                                                     spike_ids,
                                                     cluster_ids)

        # Fetch the waveforms.
        w = self.waveforms[spike_ids]
        t = _get_linear_x(n_spikes, self.n_samples)
        # Overlap.
        if not self.overlap:
            t = t + 2.5 * (spike_clusters_rel[:, np.newaxis] -
                           (n_clusters - 1) / 2.)
            # The total width should not depend on the number of clusters.
            t /= n_clusters

        # Depth as a function of the cluster index and masks.
        masks = self.masks[spike_ids]

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
                                   n_clusters=n_clusters)
                self[ch].plot(x=t, y=w[:, :, ch],
                              color=color,
                              depth=depth,
                              data_bounds=self.data_bounds,
                              )

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(WaveformView, self).attach(gui)
        self.actions.add(self.toggle_waveform_overlap)

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

    def toggle_waveform_overlap(self):
        self.overlap = not self.overlap
        self.on_select()

    # Box scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.boxed.update_boxes(self.box_pos * self.probe_scaling,
                                self.box_size * self.box_scaling)

    def widen(self):
        self.box_scaling[0] *= self.box_coeff
        self._update_boxes()

    def narrow(self):
        self.box_scaling[0] /= self.box_coeff
        self._update_boxes()

    def increase(self):
        self.box_scaling[1] *= self.box_coeff
        self._update_boxes()

    def decrease(self):
        self.box_scaling[1] /= self.box_coeff
        self._update_boxes()

    # Probe scaling
    # -------------------------------------------------------------------------

    def extend_horizontally(self):
        self.probe_scaling[0] *= self.box_coeff
        self._update_boxes()

    def shrink_horizontally(self):
        self.probe_scaling[0] /= self.box_coeff
        self._update_boxes()

    def extend_vertically(self):
        self.probe_scaling[1] *= self.box_coeff
        self._update_boxes()

    def shrink_vertically(self):
        self.probe_scaling[1] /= self.box_coeff
        self._update_boxes()


class WaveformViewPlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):
        bs, ps = state.get_view_params('WaveformView', 'box_scaling',
                                       'probe_scaling')
        view = WaveformView(waveforms=model.waveforms,
                            masks=model.masks,
                            spike_clusters=model.spike_clusters,
                            channel_positions=model.channel_positions,
                            box_scaling=bs,
                            probe_scaling=ps,
                            )
        view.attach(gui)

        @gui.connect_
        def on_close():
            # Save the box bounds.
            state.set_view_params(view,
                                  box_scaling=tuple(view.box_scaling),
                                  probe_scaling=tuple(view.probe_scaling))


# -----------------------------------------------------------------------------
# Trace view
# -----------------------------------------------------------------------------

class TraceView(ManualClusteringView):
    interval_duration = .5  # default duration of the interval
    shift_amount = .1
    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
    }

    def __init__(self,
                 traces=None,
                 sample_rate=None,
                 spike_times=None,
                 spike_clusters=None,
                 masks=None,
                 n_samples_per_spike=None,
                 **kwargs):

        # Sample rate.
        assert sample_rate > 0
        self.sample_rate = sample_rate
        self.dt = 1. / self.sample_rate

        # Traces.
        assert len(traces.shape) == 2
        self.n_samples, self.n_channels = traces.shape
        self.traces = traces
        self.duration = self.dt * self.n_samples

        # Number of samples per spike.
        self.n_samples_per_spike = (n_samples_per_spike or
                                    round(.002 * sample_rate))

        # Spike times.
        if spike_times is not None:
            spike_times = np.asarray(spike_times)
            self.n_spikes = len(spike_times)
            assert spike_times.shape == (self.n_spikes,)
            self.spike_times = spike_times

            # Spike clusters.
            spike_clusters = (np.zeros(self.n_spikes) if spike_clusters is None
                              else spike_clusters)
            assert spike_clusters.shape == (self.n_spikes,)
            self.spike_clusters = spike_clusters

            # Masks.
            assert masks.shape == (self.n_spikes, self.n_channels)
            self.masks = masks
        else:
            self.spike_times = self.spike_clusters = self.masks = None

        # Initialize the view.
        super(TraceView, self).__init__(layout='stacked',
                                        n_plots=self.n_channels,
                                        **kwargs)

        # Initial interval.
        self.set_interval((0., self.interval_duration))

    def _load_traces(self, interval):
        """Load traces in an interval (in seconds)."""

        start, end = interval

        i, j = round(self.sample_rate * start), round(self.sample_rate * end)
        i, j = int(i), int(j)
        traces = self.traces[i:j, :]

        # Detrend the traces.
        m = np.mean(traces[::10, :], axis=0).astype(traces.dtype)
        traces -= m

        # Create the plots.
        return traces

    def _load_spikes(self, interval):
        """Return spike times, spike clusters, masks."""
        assert self.spike_times is not None
        # Keep the spikes in the interval.
        a, b = self.spike_times.searchsorted(interval)
        return self.spike_times[a:b], self.spike_clusters[a:b], self.masks[a:b]

    def _plot_traces(self, traces, start=None, data_bounds=None):
        t = start + np.arange(traces.shape[0]) * self.dt
        gray = .4
        for ch in range(self.n_channels):
            self[ch].plot(t, traces[:, ch],
                          color=(gray, gray, gray, 1),
                          data_bounds=data_bounds)

    def _plot_spike(self, spike_idx, start=None,
                    traces=None, spike_times=None, spike_clusters=None,
                    masks=None, data_bounds=None):

        wave_len = self.n_samples_per_spike
        dur_spike = wave_len * self.dt
        trace_start = round(self.sample_rate * start)

        # Find the first x of the spike, relative to the start of
        # the interval
        sample_rel = (round(spike_times[spike_idx] * self.sample_rate) -
                      trace_start)

        # Extract the waveform from the traces.
        w, ch = _extract_wave(traces, sample_rel,
                              masks[spike_idx], wave_len)

        # Determine the color as a function of the spike's cluster.
        clu = spike_clusters[spike_idx]
        if self.cluster_ids is None or clu not in self.cluster_ids:
            gray = .9
            color = (gray, gray, gray, 1)
        else:
            clu_rel = self.cluster_ids.index(clu)
            r, g, b = (_COLORMAP[clu_rel % len(_COLORMAP)] / 255.)
            color = (r, g, b, 1.)
            sc = clu_rel * np.ones(len(ch), dtype=np.int32)
            color = _get_color(masks[spike_idx, ch],
                               spike_clusters_rel=sc,
                               n_clusters=len(self.cluster_ids))

        # Generate the x coordinates of the waveform.
        t0 = spike_times[spike_idx] - dur_spike / 2.
        t = t0 + self.dt * np.arange(wave_len)
        t = np.tile(t, (len(ch), 1))

        # The box index depends on the channel.
        box_index = np.repeat(ch[:, np.newaxis], wave_len, axis=0)
        self.plot(t, w.T, color=color, box_index=box_index,
                  data_bounds=data_bounds)

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

    def set_interval(self, interval):
        """Display the traces and spikes in a given interval."""
        self.clear()
        interval = self._restrict_interval(interval)
        self.interval = interval
        start, end = interval

        # Load traces.
        traces = self._load_traces(interval)
        assert traces.shape[1] == self.n_channels

        # Determine the data bounds.
        m, M = traces.min(), traces.max()
        data_bounds = np.array([start, m, end, M])

        # Plot the traces.
        # OPTIM: avoid the loop and generate all channel traces in
        # one pass with NumPy (but need to set a_box_index manually too).
        self._plot_traces(traces, start=start, data_bounds=data_bounds)

        # Display the spikes.
        if self.spike_times is not None:
            # Load the spikes.
            spike_times, spike_clusters, masks = self._load_spikes(interval)

            # Plot every spike.
            for i in range(len(spike_times)):
                self._plot_spike(i,
                                 start=start,
                                 traces=traces,
                                 spike_times=spike_times,
                                 spike_clusters=spike_clusters,
                                 masks=masks,
                                 data_bounds=data_bounds,
                                 )

        self.build()
        self.update()

    def on_select(self, cluster_ids=None, spike_ids=None):
        super(TraceView, self).on_select(cluster_ids=cluster_ids,
                                         spike_ids=spike_ids)
        self.set_interval(self.interval)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(TraceView, self).attach(gui)
        self.actions.add(self.go_to, alias='tg')
        self.actions.add(self.shift, alias='ts')
        self.actions.add(self.go_right)
        self.actions.add(self.go_left)

    def go_to(self, time):
        start, end = self.interval
        half_dur = (end - start) * .5
        self.set_interval((time - half_dur, time + half_dur))

    def shift(self, delay):
        time = sum(self.interval) * .5
        self.go_to(time + delay)

    def go_right(self):
        start, end = self.interval
        delay = (end - start) * .2
        self.shift(delay)

    def go_left(self):
        start, end = self.interval
        delay = (end - start) * .2
        self.shift(-delay)


class TraceViewPlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):
        view = TraceView(traces=model.traces,
                         sample_rate=model.sample_rate,
                         spike_times=model.spike_times,
                         spike_clusters=model.spike_clusters,
                         masks=model.masks,
                         )
        b, = state.get_view_params('TraceView', 'box_size')
        if b:
            view.stacked.box_size = b
        view.attach(gui)

        @gui.connect_
        def on_close():
            # Save the box bounds.
            state.set_view_params(view, box_size=view.stacked.box_size)


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------

def _dimensions_matrix(x_channels, y_channels):
    """Dimensions matrix."""
    # time, depth     time,    (x, 0)     time,    (y, 0)     time, (z, 0)
    # time, (x', 0)   (x', 0), (x, 0)     (x', 1), (y, 0)     (x', 2), (z, 0)
    # time, (y', 0)   (y', 0), (x, 1)     (y', 1), (y, 1)     (y', 2), (z, 1)
    # time, (z', 0)   (z', 0), (x, 2)     (z', 1), (y, 2)     (z', 2), (z, 2)

    n = len(x_channels)
    assert len(y_channels) == n
    y_dim = {}
    x_dim = {}
    # TODO: extra feature like probe depth
    x_dim[0, 0] = 'time'
    y_dim[0, 0] = 'time'

    # Time in first column and first row.
    for i in range(1, n + 1):
        x_dim[0, i] = 'time'
        y_dim[0, i] = (x_channels[i - 1], 0)
        x_dim[i, 0] = 'time'
        y_dim[i, 0] = (y_channels[i - 1], 0)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            x_dim[i, j] = (x_channels[i - 1], j - 1)
            y_dim[i, j] = (y_channels[j - 1], i - 1)

    return x_dim, y_dim


def _dimensions_for_clusters(cluster_ids, n_cols=None,
                             best_channels_func=None):
    """Return the dimension matrix for the selected clusters."""
    n = len(cluster_ids)
    if not n:
        return {}, {}
    best_channels_func = best_channels_func or (lambda _: range(n_cols))
    x_channels = best_channels_func(cluster_ids[min(1, n - 1)])
    y_channels = best_channels_func(cluster_ids[0])
    y_channels = y_channels[:n_cols - 1]
    # For the x axis, remove the channels that already are in
    # the y axis.
    x_channels = [c for c in x_channels if c not in y_channels]
    # Now, select the right number of channels in the x axis.
    x_channels = x_channels[:n_cols - 1]
    if len(x_channels) < n_cols - 1:
        x_channels = y_channels  # pragma: no cover
    return _dimensions_matrix(x_channels, y_channels)


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
    normalization_percentile = .95
    normalization_n_spikes = 1000
    _default_marker_size = 3.
    _feature_scaling = 1.

    default_shortcuts = {
        'increase_feature_scaling': 'ctrl++',
        'decrease_feature_scaling': 'ctrl+-',
    }

    def __init__(self,
                 features=None,
                 masks=None,
                 spike_times=None,
                 spike_clusters=None,
                 **kwargs):

        assert len(features.shape) == 3
        self.n_spikes, self.n_channels, self.n_features = features.shape
        self.n_cols = self.n_features + 1
        self.shape = (self.n_cols, self.n_cols)
        self.features = features

        # Initialize the view.
        super(FeatureView, self).__init__(layout='grid',
                                          shape=self.shape,
                                          **kwargs)

        # Feature normalization.
        self.data_bounds = _get_data_bounds(features,
                                            self.normalization_n_spikes,
                                            self.normalization_percentile)

        # Masks.
        self.masks = masks

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters

        # Spike times.
        assert spike_times.shape == (self.n_spikes,)

        # Attributes: extra features. This is a dictionary
        # {name: (array, data_bounds)}
        # where each array is a `(n_spikes,)` array.
        self.attributes = {}

        self.add_attribute('time', spike_times)
        self.best_channels_func = None

    def add_attribute(self, name, values):
        """Add an attribute (aka extra feature).

        The values should be a 1D array with `n_spikes` elements.

        By default, there is the `time` attribute.

        """
        assert values.shape == (self.n_spikes,)
        lim = values.min(), values.max()
        self.attributes[name] = (values, lim)

    def _get_feature(self, dim, spike_ids=None):
        f = self.features[spike_ids]
        assert f.ndim == 3

        if dim in self.attributes:
            # Extra features like time.
            values, _ = self.attributes[dim]
            values = values[spike_ids]
            assert values.shape == (len(spike_ids),)
            return values
        else:
            assert len(dim) == 2
            ch, fet = dim
            return f[:, ch, fet] * self._feature_scaling

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

    def _plot_features(self, i, j, x_dim, y_dim,
                       cluster_ids=None, spike_ids=None,
                       masks=None, spike_clusters_rel=None):
        sc = spike_clusters_rel
        n_clusters = len(cluster_ids)

        # Retrieve the x and y values for the subplot.
        x = self._get_feature(x_dim[i, j], spike_ids)
        y = self._get_feature(y_dim[i, j], spike_ids)

        # Retrieve the data bounds.
        data_bounds = self._get_dim_bounds(x_dim[i, j],
                                           y_dim[i, j])

        # Retrieve the masks and depth.
        mx, dx = _project_mask_depth(x_dim[i, j], masks,
                                     spike_clusters_rel=sc,
                                     n_clusters=n_clusters)
        my, dy = _project_mask_depth(y_dim[i, j], masks,
                                     spike_clusters_rel=sc,
                                     n_clusters=n_clusters)

        d = np.maximum(dx, dy)
        m = np.maximum(mx, my)

        # Get the color of the markers.
        color = _get_color(m,
                           spike_clusters_rel=sc,
                           n_clusters=n_clusters)

        # Create the scatter plot for the current subplot.
        ms = self._default_marker_size
        self[i, j].scatter(x=x,
                           y=y,
                           color=color,
                           depth=d,
                           data_bounds=data_bounds,
                           size=ms * np.ones(len(spike_ids)),
                           )

    def set_best_channels_func(self, func):
        """Set a function `cluster_id => list of best channels`."""
        self.best_channels_func = func

    def on_select(self, cluster_ids=None, spike_ids=None):
        super(FeatureView, self).on_select(cluster_ids=cluster_ids,
                                           spike_ids=spike_ids)
        cluster_ids, spike_ids = self.cluster_ids, self.spike_ids
        n_spikes = len(spike_ids)
        if n_spikes == 0:
            return

        # Get the masks for the selected spikes.
        masks = self.masks[spike_ids]
        sc = _get_spike_clusters_rel(self.spike_clusters,
                                     spike_ids,
                                     cluster_ids)

        f = self.best_channels_func
        x_dim, y_dim = _dimensions_for_clusters(cluster_ids,
                                                n_cols=self.n_cols,
                                                best_channels_func=f)

        # Set a non-time attribute as y coordinate in the top-left subplot.
        attrs = sorted(self.attributes)
        attrs.remove('time')
        if attrs:
            y_dim[0, 0] = attrs[0]

        # Plot all features.
        with self.building():
            for i in range(self.n_cols):
                for j in range(self.n_cols):
                    self._plot_features(i, j, x_dim, y_dim,
                                        cluster_ids=cluster_ids,
                                        spike_ids=spike_ids,
                                        masks=masks,
                                        spike_clusters_rel=sc,
                                        )
            # Add the boxes.
            self.grid.add_boxes(self, self.shape)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(FeatureView, self).attach(gui)
        self.actions.add(self.increase_feature_scaling)
        self.actions.add(self.decrease_feature_scaling)

    def increase_feature_scaling(self):
        self.feature_scaling *= 1.2
        self.on_select()

    def decrease_feature_scaling(self):
        self.feature_scaling /= 1.2
        self.on_select()

    @property
    def feature_scaling(self):
        return self._feature_scaling

    @feature_scaling.setter
    def feature_scaling(self, value):
        self._feature_scaling = value


class FeatureViewPlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):

        view = FeatureView(features=model.features,
                           masks=model.masks,
                           spike_clusters=model.spike_clusters,
                           spike_times=model.spike_times,
                           )
        fs, = state.get_view_params('FeatureView', 'feature_scaling')
        if fs:
            view.feature_scaling = fs

        @view.set_best_channels_func
        def best_channels(cluster_id):
            """Select the best channels for a given cluster."""
            # TODO: better perf with cluster stats and cache
            spike_ids = model.spikes_per_cluster[cluster_id]
            m = model.masks[spike_ids]
            mean_masks = mean(m)
            uch = unmasked_channels(mean_masks)
            return sorted_main_channels(mean_masks, uch)

        view.attach(gui)

        @gui.connect_
        def on_close():
            # Save the box bounds.
            state.set_view_params(view, feature_scaling=view.feature_scaling)


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
                 bin_size=None,
                 window_size=None,
                 excerpt_size=None,
                 n_excerpts=None,
                 **kwargs):

        assert sample_rate > 0
        self.sample_rate = sample_rate

        self.bin_size = bin_size or self.bin_size
        assert self.bin_size > 0

        self.window_size = window_size or self.window_size
        assert self.window_size > 0

        self.excerpt_size = excerpt_size or self.excerpt_size
        self.n_excerpts = n_excerpts or self.n_excerpts

        self.spike_times = np.asarray(spike_times)
        self.n_spikes, = self.spike_times.shape

        # Initialize the view.
        super(CorrelogramView, self).__init__(layout='grid',
                                              shape=(1, 1),
                                              **kwargs)

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters

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

    def on_select(self, cluster_ids=None, spike_ids=None):
        super(CorrelogramView, self).on_select(cluster_ids=cluster_ids,
                                               spike_ids=spike_ids)
        cluster_ids, spike_ids = self.cluster_ids, self.spike_ids
        n_clusters = len(cluster_ids)
        n_spikes = len(spike_ids)
        if n_spikes == 0:
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
        self.uniform_normalization = not self.uniform_normalization
        self.on_select()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(CorrelogramView, self).attach(gui)
        self.actions.add(self.toggle_normalization, shortcut='n')


class CorrelogramViewPlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):
        bs, ws, es, ne = state.get_view_params('CorrelogramView',
                                               'bin_size',
                                               'window_size',
                                               'excerpt_size',
                                               'n_excerpts',
                                               )
        view = CorrelogramView(spike_times=model.spike_times,
                               spike_clusters=model.spike_clusters,
                               sample_rate=model.sample_rate,
                               bin_size=bs,
                               window_size=ws,
                               excerpt_size=es,
                               n_excerpts=ne,
                               )
        view.attach(gui)
