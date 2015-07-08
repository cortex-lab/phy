# -*- coding: utf-8 -*-

"""Plotting waveforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo
from vispy.gloo import Texture2D

from ._panzoom import PanZoom
from ._vispy_utils import (BaseSpikeVisual,
                           BaseSpikeCanvas,
                           _enable_depth_mask,
                           _wrap_vispy,
                           )
from ..utils._types import _as_array
from ..utils._color import _selected_clusters_colors
from ..utils.array import _index_of, _normalize, _unique
from ..electrode.mea import linear_positions


#------------------------------------------------------------------------------
# Waveform visual
#------------------------------------------------------------------------------

class WaveformVisual(BaseSpikeVisual):
    """Display waveforms with probe geometry."""

    _shader_name = 'waveforms'
    _gl_draw_mode = 'line_strip'

    def __init__(self, **kwargs):
        super(WaveformVisual, self).__init__(**kwargs)

        self._waveforms = None
        self.n_channels, self.n_samples = None, None
        self._channel_order = None
        self._channel_positions = None

        self.program['u_data_scale'] = (.05, .05)
        self.program['u_channel_scale'] = (1., 1.)
        self.program['u_overlap'] = 0.
        self.program['u_alpha'] = 0.5
        _enable_depth_mask()

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def waveforms(self):
        """Displayed waveforms.

        This is a `(n_spikes, n_samples, n_channels)` array.

        """
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
        self._empty = self.n_spikes == 0
        self.set_to_bake('spikes', 'spikes_clusters', 'color')

    @property
    def channel_positions(self):
        """Positions of the channels.

        This is a `(n_channels, 2)` array.

        """
        return self._channel_positions

    @channel_positions.setter
    def channel_positions(self, value):
        value = _as_array(value)
        self._channel_positions = value
        self.set_to_bake('channel_positions')

    @property
    def channel_order(self):
        return self._channel_order

    @channel_order.setter
    def channel_order(self, value):
        self._channel_order = value

    @property
    def alpha(self):
        """Alpha transparency (between 0 and 1)."""
        return self.program['u_alpha']

    @alpha.setter
    def alpha(self, value):
        self.program['u_alpha'] = value

    @property
    def box_scale(self):
        """Scale of the waveforms.

        This is a pair of scalars.

        """
        return tuple(self.program['u_data_scale'])

    @box_scale.setter
    def box_scale(self, value):
        assert len(value) == 2
        self.program['u_data_scale'] = value

    @property
    def probe_scale(self):
        """Scale of the probe.

        This is a pair of scalars.

        """
        return tuple(self.program['u_channel_scale'])

    @probe_scale.setter
    def probe_scale(self, value):
        assert len(value) == 2
        self.program['u_channel_scale'] = value

    @property
    def overlap(self):
        """Whether to overlap waveforms."""
        return True if self.program['u_overlap'][0] > .5 else False

    @overlap.setter
    def overlap(self, value):
        assert value in (True, False)
        self.program['u_overlap'] = 1. if value else 0.

    def channel_hover(self, position):
        """Return the channel id closest to the mouse pointer.

        Parameters
        ----------

        position : tuple
            The normalized coordinates of the mouse pointer, in world
            coordinates (in `[-1, 1]`).

        """
        mouse_pos = position / self.probe_scale
        # Normalize channel positions.
        positions = self.channel_positions.astype(np.float32)
        positions = _normalize(positions, keep_ratio=True)
        positions = .1 + .8 * positions
        positions = 2 * positions - 1
        # Find closest channel.
        d = np.sum((positions - mouse_pos[None, :]) ** 2, axis=1)
        idx = np.argmin(d)
        # if self.channel_order is not None:
        #     channel_id = self.channel_order[idx]
        # WARNING: by convention this is the relative channel index.
        return idx

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
        self.program['u_channel_pos'] = Texture2D(u_channel_pos,
                                                  wrapping='clamp_to_edge')

    def _bake_spikes(self):

        # Bake masks.
        # WARNING: swap channel/time axes in the waveforms array.
        waveforms = np.swapaxes(self._waveforms, 1, 2)
        assert waveforms.shape == (self.n_spikes,
                                   self.n_channels,
                                   self.n_samples,
                                   )
        masks = np.repeat(self._masks.ravel(), self.n_samples)
        data = np.c_[waveforms.ravel(), masks.ravel()].astype(np.float32)
        # TODO: more efficient to update the data from an existing VBO
        self.program['a_data'] = data

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
        self.program['n_channels'] = self.n_channels

    def _bake_spikes_clusters(self):
        # WARNING: needs to be called *after* _bake_spikes().
        if not hasattr(self, '_n_channels_per_spike'):
            raise RuntimeError("'_bake_spikes()' needs to be called before "
                               "'bake_spikes_clusters().")
        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters
        # We take the cluster order into account here.
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_order)
        # Generate the box attribute.
        assert len(spike_clusters_idx) == len(self._n_channels_per_spike)
        a_cluster = np.repeat(spike_clusters_idx,
                              self._n_channels_per_spike * self.n_samples)
        a_channel = np.repeat(self._channels_per_spike, self.n_samples)
        a_box = np.c_[a_cluster, a_channel].astype(np.float32)
        # TODO: more efficient to update the data from an existing VBO
        self.program['a_box'] = a_box
        self.program['n_clusters'] = self.n_clusters


class WaveformView(BaseSpikeCanvas):
    """A VisPy canvas displaying waveforms."""
    _visual_class = WaveformVisual
    _arrows = ('Left', 'Right', 'Up', 'Down')
    _pm = ('+', '-')
    _events = ('channel_click',)
    _key_pressed = None
    _show_mean = False

    def _create_visuals(self):
        super(WaveformView, self)._create_visuals()
        self.mean = WaveformVisual()
        self.mean.alpha = 1.

    def _create_pan_zoom(self):
        self._pz = PanZoom()
        self._pz.add(self.visual.program)
        self._pz.add(self.mean.program)
        self._pz.attach(self)

    def set_data(self,
                 waveforms=None,
                 masks=None,
                 spike_clusters=None,
                 channel_positions=None,
                 channel_order=None,
                 colors=None,
                 **kwargs
                 ):

        if waveforms is not None:
            assert isinstance(waveforms, np.ndarray)
            if waveforms.ndim == 2:
                waveforms = waveforms[None, ...]
            assert waveforms.ndim == 3
        else:
            waveforms = self.visual.waveforms
        n_spikes, n_samples, n_channels = waveforms.shape

        if spike_clusters is None:
            spike_clusters = np.zeros(n_spikes, dtype=np.int32)
        spike_clusters = _as_array(spike_clusters)
        cluster_ids = _unique(spike_clusters)
        n_clusters = len(cluster_ids)

        if masks is None:
            masks = np.ones((n_spikes, n_channels), dtype=np.float32)

        if colors is None:
            colors = _selected_clusters_colors(n_clusters)

        if channel_order is None:
            channel_order = self.visual.channel_order
        if channel_order is None:
            channel_order = np.arange(n_channels)

        if channel_positions is None:
            channel_positions = self.visual.channel_positions
        if channel_positions is None:
            channel_positions = linear_positions(n_channels)

        self.visual.waveforms = waveforms.astype(np.float32)

        if masks is not None:
            self.visual.masks = masks

        self.visual.spike_clusters = spike_clusters
        assert spike_clusters.shape == (n_spikes,)

        self.visual.cluster_colors = colors
        self.visual.channel_positions = channel_positions
        self.visual.channel_order = channel_order

        # Extra parameters.
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.update()

    @property
    def box_scale(self):
        """Scale of the waveforms.

        This is a pair of scalars.

        """
        return self.visual.box_scale

    @box_scale.setter
    def box_scale(self, value):
        self.visual.box_scale = value
        self.mean.box_scale = value
        self.update()

    @property
    def probe_scale(self):
        """Scale of the probe.

        This is a pair of scalars.

        """
        return self.visual.probe_scale

    @probe_scale.setter
    def probe_scale(self, value):
        self.visual.probe_scale = value
        self.mean.probe_scale = value
        self.update()

    @property
    def alpha(self):
        """Opacity."""
        return self.visual.alpha

    @alpha.setter
    def alpha(self, value):
        self.visual.alpha = value
        self.mean.alpha = value
        self.update()

    @property
    def overlap(self):
        """Whether to overlap waveforms."""
        return self.visual.overlap

    @overlap.setter
    def overlap(self, value):
        self.visual.overlap = value
        self.mean.overlap = value
        self.update()

    @property
    def show_mean(self):
        """Whether to show_mean waveforms."""
        return self._show_mean

    @show_mean.setter
    def show_mean(self, value):
        self._show_mean = value
        self.update()

    keyboard_shortcuts = {
        'waveform_scale_increase': ('ctrl+',
                                    'ctrl+up',
                                    'shift+wheel up',
                                    ),
        'waveform_scale_decrease': ('ctrl-',
                                    'ctrl+down',
                                    'shift+wheel down',
                                    ),
        'waveform_width_increase': ('ctrl+right', 'ctrl+wheel up'),
        'waveform_width_decrease': ('ctrl+left', 'ctrl+wheel down'),
        'probe_width_increase': ('shift+right', 'ctrl+alt+wheel up'),
        'probe_width_decrease': ('shift+left', 'ctrl+alt+wheel down'),
        'probe_height_increase': ('shift+up', 'shift+alt+wheel up'),
        'probe_height_decrease': ('shift+down', 'shift+alt+wheel down'),
        'select_channel': ('ctrl+left click', 'ctrl+right click', 'num+click'),
    }

    def on_key_press(self, event):
        """Handle key press events."""
        key = event.key

        self._key_pressed = key

        ctrl = 'Control' in event.modifiers
        shift = 'Shift' in event.modifiers

        # Box scale.
        if ctrl and key in self._arrows + self._pm:
            coeff = 1.1
            u, v = self.box_scale
            if key == 'Left':
                self.box_scale = (u / coeff, v)
            elif key == 'Right':
                self.box_scale = (u * coeff, v)
            elif key in ('Down', '-'):
                self.box_scale = (u, v / coeff)
            elif key in ('Up', '+'):
                self.box_scale = (u, v * coeff)

        # Probe scale.
        if shift and key in self._arrows:
            coeff = 1.1
            u, v = self.probe_scale
            if key == 'Left':
                self.probe_scale = (u / coeff, v)
            elif key == 'Right':
                self.probe_scale = (u * coeff, v)
            elif key == 'Down':
                self.probe_scale = (u, v / coeff)
            elif key == 'Up':
                self.probe_scale = (u, v * coeff)

    def on_key_release(self, event):
        self._key_pressed = None

    def on_mouse_wheel(self, event):
        """Handle mouse wheel events."""
        ctrl = 'Control' in event.modifiers
        shift = 'Shift' in event.modifiers
        alt = 'Alt' in event.modifiers
        coeff = 1. + .1 * event.delta[1]

        # Box scale.
        if ctrl and not alt:
            u, v = self.box_scale
            self.box_scale = (u * coeff, v)
        if shift and not alt:
            u, v = self.box_scale
            self.box_scale = (u, v * coeff)

        # Probe scale.
        if ctrl and alt:
            u, v = self.probe_scale
            self.probe_scale = (u * coeff, v)
        if shift and alt:
            u, v = self.probe_scale
            self.probe_scale = (u, v * coeff)

    def on_mouse_press(self, e):
        key = self._key_pressed
        if 'Control' in e.modifiers or key in map(str, range(10)):
            box_idx = int(key.name) if key in map(str, range(10)) else None
            # Normalise mouse position.
            position = self._pz._normalize(e.pos)
            position[1] = -position[1]
            zoom = self._pz._zoom_aspect()
            pan = self._pz.pan
            mouse_pos = ((position / zoom) - pan)
            # Find the channel id.
            channel_idx = self.visual.channel_hover(mouse_pos)
            self.emit("channel_click",
                      channel_idx=channel_idx,
                      ax='x' if e.button == 1 else 'y',
                      box_idx=box_idx,
                      )

    def on_draw(self, event):
        """Draw the visual."""
        gloo.clear(color=True, depth=True)
        if self._show_mean:
            self.mean.draw()
        else:
            self.visual.draw()


#------------------------------------------------------------------------------
# Plotting functions
#------------------------------------------------------------------------------

@_wrap_vispy
def plot_waveforms(waveforms, **kwargs):
    """Plot waveforms.

    Parameters
    ----------

    waveforms : ndarray
        The waveforms to plot. A `(n_spikes, n_samples, n_channels)` array.
    spike_clusters : ndarray (optional)
        A `(n_spikes,)` int array with the spike clusters.
    masks : ndarray (optional)
        A `(n_spikes, n_channels)` float array with the spike masks.
    channel_positions : ndarray
        A `(n_channels, 2)` array with the channel positions.

    """
    c = WaveformView(keys='interactive')
    c.set_data(waveforms, **kwargs)
    return c
