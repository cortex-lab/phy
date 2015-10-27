# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.io.array import _index_of
from phy.electrode.mea import linear_positions
from phy.plot import BoxedView, StackedView, GridView, _get_linear_x
from phy.plot.utils import _get_boxes, _get_array

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


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

class WaveformView(BoxedView):
    def __init__(self,
                 waveforms=None,
                 masks=None,
                 spike_clusters=None,
                 channel_positions=None,
                 ):
        """

        The channel order in waveforms needs to correspond to the one
        in channel_positions.

        """

        # Initialize the view.
        if channel_positions is None:
            channel_positions = linear_positions(self.n_channels)
        box_bounds = _get_boxes(channel_positions)
        super(WaveformView, self).__init__(box_bounds)

        # Waveforms.
        assert waveforms.ndim == 3
        self.n_spikes, self.n_samples, self.n_channels = waveforms.shape
        self.waveforms = waveforms

        # Masks.
        self.masks = _get_array(masks, (self.n_spikes, self.n_channels), 1)

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters

        # Channel positions.
        assert channel_positions.shape == (self.n_channels, 2)
        self.channel_positions = channel_positions

    def on_select(self, cluster_ids, spike_ids):
        n_clusters = len(cluster_ids)
        n_spikes = len(spike_ids)
        if n_spikes == 0:
            return

        # Relative spike clusters.
        # NOTE: the order of the clusters in cluster_ids matters.
        # It will influence the relative index of the clusters, which
        # in return influence the depth.
        spike_clusters = self.spike_clusters[spike_ids]
        assert np.all(np.in1d(spike_clusters, cluster_ids))
        spike_clusters_rel = _index_of(spike_clusters, cluster_ids)

        # Fetch the waveforms.
        w = self.waveforms[spike_ids]
        colors = _selected_clusters_colors(n_clusters)
        t = _get_linear_x(n_spikes, self.n_samples)

        # Get the colors.
        color = colors[spike_clusters_rel]
        # Alpha channel.
        color = np.c_[color, np.ones((n_spikes, 1))]

        # Depth as a function of the cluster index and masks.
        m = self.masks[spike_ids, :]
        depth = -0.1 - (spike_clusters_rel[:, np.newaxis] + m)
        assert depth.shape == (n_spikes, self.n_channels)
        depth = depth / float(n_clusters + 10.)
        depth[m <= 0.25] = 0

        # Plot all waveforms.
        for ch in range(self.n_channels):
            self[ch].plot(x=t, y=w[:, :, ch],
                          color=color,
                          depth=depth[:, ch])

        self.build()
        self.update()

    def on_cluster(self, up):
        pass

    def on_mouse_move(self, e):
        pass

    def on_key_press(self, e):
        pass

    def attach_to_gui(self, gui):
        gui.add_view(self)

        # TODO: make sure the GUI emits these events
        gui.connect(self.on_select)
        gui.connect(self.on_cluster)


class TraceView(StackedView):
    def __init__(self,
                 traces=None,
                 sample_rate=None,
                 spike_times=None,
                 spike_clusters=None,
                 masks=None,
                 n_samples_per_spike=None,
                 ):

        # Sample rate.
        assert sample_rate > 0
        self.sample_rate = sample_rate

        # Traces.
        assert traces.ndim == 2
        self.n_samples, self.n_channels = traces.shape
        self.traces = traces

        # Number of samples per spike.
        self.n_samples_per_spike = (n_samples_per_spike or
                                    int(.002 * sample_rate))

        # Spike times.
        if spike_times is not None:
            self.n_spikes = len(spike_times)
            assert spike_times.shape == (self.n_spikes,)
            self.spike_times = spike_times

            # Spike clusters.
            if spike_clusters is None:
                spike_clusters = np.zeros(self.n_spikes)
            assert spike_clusters.shape == (self.n_spikes,)
            self.spike_clusters = spike_clusters

            # Masks.
            masks = _get_array(masks, (self.n_spikes, self.n_channels), 1)
            assert masks.shape == (self.n_spikes, self.n_channels)
            self.masks = masks
        else:
            self.spike_times = self.spike_clusters = self.masks = None

        # Initialize the view.
        super(TraceView, self).__init__(self.n_channels)

        # TODO: choose the interval.
        self.set_interval((0., .25))

    def _load_traces(self, interval):
        """Load traces in an interval (in seconds)."""

        start, end = interval

        i, j = int(self.sample_rate * start), int(self.sample_rate * end)
        traces = self.traces[i:j, :]

        # Detrend the traces.
        m = np.mean(traces[::10, :], axis=0)
        traces -= m

        # Create the plots.
        return traces

    def _load_spikes(self, interval):
        assert self.spike_times is not None
        # Keep the spikes in the interval.
        a, b = self.spike_times.searchsorted(interval)
        return self.spike_times[a:b], self.spike_clusters[a:b], self.masks[a:b]

    def set_interval(self, interval):

        color = (.5, .5, .5, 1)

        # Load traces.
        traces = self._load_traces(interval)
        assert traces.shape[1] == self.n_channels

        # Generate the trace plots.
        # TODO OPTIM: avoid the loop and generate all channel traces in
        # one pass with NumPy (but need to set a_box_index manually too).
        t = _get_linear_x(1, traces.shape[0])
        for ch in range(self.n_channels):
            self[ch].plot(t, traces[:, ch], color=color)

        # if self.spike_times is not None:
        #     spike_times, spike_clusters, masks = self._load_spikes(interval)

        self.build()
        self.update()

    # # Keep the spikes in the interval.
    # spikes = self.spike_ids
    # spike_samples = self.model.spike_samples[spikes]
    # a, b = spike_samples.searchsorted(interval)
    # spikes = spikes[a:b]
    # self.view.visual.n_spikes = len(spikes)
    # self.view.visual.spike_ids = spikes


    # def on_select(self, cluster_ids, spike_ids):
        # n_clusters = len(cluster_ids)
        # n_spikes = len(spike_ids)
        # if n_spikes == 0:
        #     return

        # # Relative spike clusters.
        # # NOTE: the order of the clusters in cluster_ids matters.
        # # It will influence the relative index of the clusters, which
        # # in return influence the depth.
        # spike_clusters = self.spike_clusters[spike_ids]
        # assert np.all(np.in1d(spike_clusters, cluster_ids))
        # spike_clusters_rel = _index_of(spike_clusters, cluster_ids)

        # # Fetch the waveforms.
        # w = self.waveforms[spike_ids]
        # colors = _selected_clusters_colors(n_clusters)
        # t = _get_linear_x(n_spikes, self.n_samples)

        # # Get the colors.
        # color = colors[spike_clusters_rel]
        # # Alpha channel.
        # color = np.c_[color, np.ones((n_spikes, 1))]

        # # Plot all waveforms.
        # for ch in range(self.n_channels):
        #     self[ch].plot(x=t, y=w[:, :, ch],
        #                   color=color,
        #                   depth=depth[:, ch])

        # self.build()
        # self.update()


class FeatureView(GridView):
    def __init__(self,
                 features=None,
                 dimensions=None,
                 extra_features=None,
                 ):
        pass


class CorrelogramView(GridView):
    def __init__(self,
                 spike_samples=None,
                 spike_times=None,
                 bin_size=None,
                 window_size=None,
                 excerpt_size=None,
                 n_excerpts=None,
                 ):
        pass
