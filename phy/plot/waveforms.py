# -*- coding: utf-8 -*-

"""Plotting waveforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo

from ._vispy_utils import BaseSpikeVisual, BaseSpikeCanvas
from ..utils.array import _as_array, _index_of, _normalize
from ..utils.logging import debug
from ..utils._color import _random_color


#------------------------------------------------------------------------------
# Waveform visual
#------------------------------------------------------------------------------

class WaveformVisual(BaseSpikeVisual):

    _shader_name = 'waveforms'
    _gl_draw_mode = 'line_strip'

    """Waveform visual."""
    def __init__(self, **kwargs):
        super(WaveformVisual, self).__init__(**kwargs)

        self._waveforms = None
        self.n_channels, self.n_samples = None, None

        self.program['u_data_scale'] = (.05, .03)

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def waveforms(self):
        """Displayed waveforms."""
        return self._waveforms

    @waveforms.setter
    def waveforms(self, value):
        # WARNING: when setting new data, waveforms need to be set first.
        # n_spikes will be set as a function of waveforms.
        value = _as_array(value)
        # TODO: support sparse structures
        assert value.ndim == 3
        self.n_spikes, self.n_samples, self.n_channels = value.shape
        self._waveforms = value
        self.set_to_bake('spikes', 'spikes_clusters', 'color')

    @property
    def channel_positions(self):
        """Array with the coordinates of all channels."""
        return self._channel_positions

    @channel_positions.setter
    def channel_positions(self, value):
        value = _as_array(value)
        self._channel_positions = value
        self.set_to_bake('channel_positions')

    @property
    def box_scale(self):
        return tuple(self.program['u_data_scale'])

    @box_scale.setter
    def box_scale(self, value):
        assert isinstance(value, tuple) and len(value) == 2
        self.program['u_data_scale'] = value
        self.update()

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_channel_positions(self):
        # WARNING: channel_positions must be in [0,1] because we have a
        # texture.
        positions = self.channel_positions.astype(np.float32)
        positions = _normalize(positions, keep_ratio=True)
        positions = positions.reshape((1, self.n_channels, -1))
        # Rescale a bit and recenter.
        positions = .1 + .8 * positions
        u_channel_pos = np.dstack((positions,
                                  np.zeros((1, self.n_channels, 1))))
        u_channel_pos = (u_channel_pos * 255).astype(np.uint8)
        # TODO: more efficient to update the data from an existing texture
        self.program['u_channel_pos'] = gloo.Texture2D(u_channel_pos,
                                                       wrapping='clamp_to_edge')  # noqa
        debug("bake channel pos", u_channel_pos.shape)

    def _bake_spikes(self):

        # Bake masks.
        # WARNING: swap channel/time axes in the waveforms array.
        waveforms = np.swapaxes(self._waveforms, 1, 2)
        masks = np.repeat(self._masks.ravel(), self.n_samples)
        data = np.c_[waveforms.ravel(), masks.ravel()].astype(np.float32)
        # TODO: more efficient to update the data from an existing VBO
        self.program['a_data'] = data
        debug("bake spikes", data.shape)

        # TODO: SparseCSR, this should just be 'channel'
        self._channels_per_spike = np.tile(np.arange(self.n_channels).
                                           astype(np.float32),
                                           self.n_spikes)

        # TODO: SparseCSR, this should be np.diff(spikes_ptr)
        self._n_channels_per_spike = self.n_channels * np.ones(self.n_spikes,
                                                               dtype=np.int32)

        self._n_waveforms = np.sum(self._n_channels_per_spike)

        # TODO: precompute this with a maximum number of waveforms?
        a_time = np.tile(np.linspace(-1., 1., self.n_samples),
                         self._n_waveforms).astype(np.float32)

        self.program['a_time'] = a_time
        self.program['n_clusters'] = self.n_clusters
        self.program['n_channels'] = self.n_channels

    def _bake_spikes_clusters(self):
        # WARNING: needs to be called *after* _bake_spikes().
        if not hasattr(self, '_n_channels_per_spike'):
            raise RuntimeError("'_bake_spikes()' needs to be called before "
                               "'bake_spikes_clusters().")
        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters[self.spike_ids]
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_ids)
        # Generate the box attribute.
        a_cluster = np.repeat(spike_clusters_idx,
                              self._n_channels_per_spike * self.n_samples)
        a_channel = np.repeat(self._channels_per_spike, self.n_samples)
        a_box = np.c_[a_cluster, a_channel].astype(np.float32)
        # TODO: more efficient to update the data from an existing VBO
        self.program['a_box'] = a_box
        debug("bake spikes clusters", a_box.shape)


class WaveformView(BaseSpikeCanvas):
    _visual_class = WaveformVisual

    def on_key_press(self, event):
        # TODO: more interactivity
        # TODO: keyboard shortcut manager
        # super(WaveformView, self).on_key_press(event)
        u, v = self.visual.box_scale
        coeff = 1.1
        if event.key == '+':
            if 'Control' in event.modifiers:
                self.visual.box_scale = (u*coeff, v)
            else:
                self.visual.box_scale = (u, v*coeff)
        if event.key == '-':
            if 'Control' in event.modifiers:
                self.visual.box_scale = (u/coeff, v)
            else:
                self.visual.box_scale = (u, v/coeff)


def add_waveform_view(session, backend=None):
    """Add a waveform view in a session.

    This function binds the session events to the created waveform view.

    The caller needs to show the waveform view explicitly.

    """
    if backend in ('pyqt4', None):
        kwargs = {'always_on_top': True}
    else:
        kwargs = {}
    view = WaveformView(**kwargs)

    @session.connect
    def on_open():
        if session.model is None:
            return
        view.visual.spike_clusters = session.clustering.spike_clusters
        view.visual.channel_positions = session.model.probe.positions
        view.update()

    @session.connect
    def on_cluster(up=None):
        pass
        # TODO: select the merged cluster
        # session.select(merged)

    @session.connect
    def on_select(selector):
        spikes = selector.selected_spikes
        if len(spikes) == 0:
            return
        if view.visual.spike_clusters is None:
            on_open()
        view.visual.waveforms = session.model.waveforms[spikes]
        view.visual.masks = session.model.masks[spikes]
        view.visual.spike_ids = spikes
        # TODO: how to choose cluster colors?
        view.visual.cluster_colors = [_random_color()
                                      for _ in selector.selected_clusters]
        view.update()

    # Unregister the callbacks when the view is closed.
    @view.connect
    def on_close(event):
        session.unconnect(on_open, on_cluster, on_select)

    # TODO: first_draw() event in VisPy view that is emitted when the view
    # is first rendered (first paint event).
    @view.connect
    def on_draw(event):
        if view.visual.spike_clusters is None:
            on_open()
            on_select(session.selector)

    return view
