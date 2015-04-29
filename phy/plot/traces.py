# -*- coding: utf-8 -*-

"""Plotting traces."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo

from ._vispy_utils import (BaseSpikeVisual,
                           BaseSpikeCanvas,
                           )
from ..utils.array import _as_array, _index_of
from ..utils.logging import debug


#------------------------------------------------------------------------------
# CCG visual
#------------------------------------------------------------------------------

class TraceVisual(BaseSpikeVisual):

    _shader_name = 'traces'
    _gl_draw_mode = 'line_strip'
    default_channel_scale = 1.

    """TraceVisual visual."""
    def __init__(self, **kwargs):
        super(TraceVisual, self).__init__(**kwargs)
        self._traces = None
        self._spike_samples = None
        self._n_samples_per_spike = None
        self._sample_rate = None
        self._offset = None

        self.program['u_scale'] = self.default_channel_scale

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def traces(self):
        """Displayed traces."""
        return self._traces

    @traces.setter
    def traces(self, value):
        value = _as_array(value)
        assert value.ndim == 2
        self.n_samples, self.n_channels = value.shape
        self._traces = value
        self._empty = self.n_samples == 0
        self._channel_colors = .5 * np.ones((self.n_channels, 3),
                                            dtype=np.float32)
        self.set_to_bake('traces', 'channel_color')

    @property
    def channel_colors(self):
        """Colors of the displayed channels."""
        return self._channel_colors

    @channel_colors.setter
    def channel_colors(self, value):
        self._channel_colors = _as_array(value)
        assert len(self._channel_colors) == self.n_channels
        self.set_to_bake('channel_color')

    @property
    def spike_samples(self):
        return self._spike_samples

    @spike_samples.setter
    def spike_samples(self, value):
        assert isinstance(value, np.ndarray)
        self._set_or_assert_n_spikes(value)
        self._spike_samples = value
        self.set_to_bake('spikes')

    @property
    def n_samples_per_spike(self):
        return self._n_samples_per_spike

    @n_samples_per_spike.setter
    def n_samples_per_spike(self, value):
        self._n_samples_per_spike = int(value)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = int(value)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = int(value)

    @property
    def channel_scale(self):
        return self.program['u_scale']

    @channel_scale.setter
    def channel_scale(self, value):
        self.program['u_scale'] = value

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_traces(self):
        ns, nc = self.n_samples, self.n_channels

        a_index = np.empty((nc * ns, 2), dtype=np.float32)
        a_index[:, 0] = np.repeat(np.arange(nc), ns)
        a_index[:, 1] = np.tile(np.arange(ns), nc)

        self.program['a_position'] = self._traces.T.ravel().astype(np.float32)
        self.program['a_index'] = a_index
        self.program['n_channels'] = nc
        self.program['n_samples'] = ns

        debug("bake traces", self._traces.shape)

    def _bake_channel_color(self):
        u_channel_color = self._channel_colors.reshape((1,
                                                        self.n_channels,
                                                        -1))
        u_channel_color = (u_channel_color * 255).astype(np.uint8)
        self.program['u_channel_color'] = gloo.Texture2D(u_channel_color)

        debug("bake channel color", u_channel_color.shape)

    def _bake_spikes(self):
        # Handle the case where there are no spikes.
        if self.n_spikes == 0:
            a_spike = np.zeros((self.n_channels * self.n_samples, 2),
                               dtype=np.float32)
            a_spike[:, 0] = -1.
            self.program['a_spike'] = a_spike
            return

        spike_clusters_idx = self.spike_clusters
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_ids)
        assert spike_clusters_idx.shape == (self.n_spikes,)

        samples = self._spike_samples
        assert samples.shape == (self.n_spikes,)

        # -1 = there's no spike at this vertex
        a_clusters = np.empty((self.n_channels, self.n_samples),
                              dtype=np.float32)
        a_clusters.fill(-1.)
        a_masks = np.zeros((self.n_channels, self.n_samples),
                           dtype=np.float32)
        masks = self._masks.T
        # Set the spike clusters and masks of all spikes, for every waveform
        # sample shift.
        for i in range(-self._n_samples_per_spike // 2,
                       +self._n_samples_per_spike // 2):

            ind = (samples + i).astype(np.uint64)
            assert ind.shape == (self.n_spikes,)

            # Skip vertices for spikes crossing the boundaries.
            u = np.nonzero(ind >= self.n_samples)[0]
            if len(u) > 0:
                u = u[0]
                if u == 0:
                    continue
                ind = ind[:u]
                spike_clusters_idx = spike_clusters_idx[:u]
                masks = masks[:, :u]

            a_clusters[:, ind] = spike_clusters_idx
            a_masks[:, ind] = masks

        a_spike = np.empty((self.n_channels * self.n_samples, 2),
                           dtype=np.float32)
        a_spike[:, 0] = a_clusters.ravel()
        a_spike[:, 1] = a_masks.ravel()
        assert a_spike.dtype == np.float32
        self.program['a_spike'] = a_spike
        self.program['n_clusters'] = self.n_clusters

    def _bake_spikes_clusters(self):
        self._bake_spikes()


class TraceView(BaseSpikeCanvas):
    """Display traces.

    Interactivity
    -------------

    Load more data:

    * Keyboard : Control and Left/Right

    Change channel scale:

    * Keyboard : Control and Up/Down

    """
    _visual_class = TraceVisual

    def __init__(self, *args, **kwargs):
        super(TraceView, self).__init__(*args, **kwargs)
        self._pz.aspect = None
        self._pz.zmin = .5
        self._pz.xmin = -1.
        self._pz.xmax = +1.
        self._pz.ymin = -2.
        self._pz.ymax = +2.

    @property
    def channel_scale(self):
        return self.visual.channel_scale

    @channel_scale.setter
    def channel_scale(self, value):
        self.visual.channel_scale = value
        self.update()

    _arrows = ('Left', 'Right', 'Up', 'Down')

    def on_key_press(self, event):

        key = event.key
        ctrl = 'Control' in event.modifiers

        # Box scale.
        if ctrl and key in self._arrows:
            coeff = 1.1
            u = self.channel_scale
            if key == 'Down':
                self.channel_scale = u / coeff
            elif key == 'Up':
                self.channel_scale = u * coeff
            self.update()
