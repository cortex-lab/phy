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

    """TraceVisual visual."""
    def __init__(self, **kwargs):
        super(TraceVisual, self).__init__(**kwargs)
        self._traces = None
        self._spike_samples = None
        self._n_samples_per_spike = None
        self._sample_rate = None
        self._offset = None

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
        assert value.shape == (self.n_spikes,)
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

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_traces(self):
        ns, nc = self.n_samples, self.n_channels

        a_index = np.c_[np.repeat(np.arange(nc), ns),
                        np.tile(np.arange(ns), nc)].astype(np.float32)

        self.program['a_position'] = self._traces.ravel().astype(np.float32)
        self.program['a_index'] = a_index
        self.program['n_channels'] = nc
        self.program['n_samples'] = ns
        self.program['u_scale'] = 1.0

        debug("bake traces", self._traces.shape)

    def _bake_channel_color(self):
        u_channel_color = self._channel_colors.reshape((1,
                                                        self.n_channels,
                                                        -1))
        u_channel_color = (u_channel_color * 255).astype(np.uint8)
        self.program['u_channel_color'] = gloo.Texture2D(u_channel_color)

        debug("bake channel color", u_channel_color.shape)

    def _bake_spikes(self):
        spike_clusters_idx = self.spike_clusters[self.spike_ids]
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_ids)
        assert spike_clusters_idx.shape == (self.n_spikes,)

        samples = self._spike_samples
        assert samples.shape == (self.n_spikes,)

        a_clusters = -np.ones((self.n_channels, self.n_samples))
        a_masks = np.zeros((self.n_channels, self.n_samples))
        masks = self._masks.T

        # Set the spike clusters and masks of all spikes, for every waveform
        # sample shift.
        for i in range(-self._n_samples_per_spike // 2,
                       +self._n_samples_per_spike // 2):

            ind = (samples + i).astype(np.uint64)
            assert ind.shape == (self.n_spikes,)

            a_clusters[:, ind] = spike_clusters_idx
            a_masks[:, ind] = masks

        a_spike = np.c_[a_clusters.ravel(),
                        a_masks.ravel()].astype(np.float32)
        self.program['a_spike'] = a_spike
        self.program['n_clusters'] = self.n_clusters


class TraceView(BaseSpikeCanvas):
    _visual_class = TraceVisual

    def __init__(self, *args, **kwargs):
        super(TraceView, self).__init__(*args, **kwargs)
        self._pz.aspect = None
        self._pz.zmin = 1
