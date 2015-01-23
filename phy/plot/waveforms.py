# -*- coding: utf-8 -*-

"""Plotting waveforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo
from vispy.gloo import Texture2D
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable
from vispy.visuals.glsl.color import HSV_TO_RGB, RGB_TO_HSV


# TODO: use ST instead of PanZoom
from ..utils.array import _unique
from ._utils import PanZoomCanvas
from ..utils.logging import debug


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

UPDATE_ORDER = {
    'spikes': 10,
    'clusters': 20,
    'cluster_metadata': 30,
}


def _check_order(changed, compare_to):
    return UPDATE_ORDER[changed] <= UPDATE_ORDER[compare_to]


#------------------------------------------------------------------------------
# Waveforms visual
#------------------------------------------------------------------------------

class Waveforms(Visual):
    VERT_SHADER = """
    // TODO: add depth
    attribute vec2 a_data;  // -1..1
    attribute float a_time;  // -1..1
    attribute vec2 a_box;  // 0..(n_clusters-1, n_channels-1)

    uniform float n_clusters;
    uniform float n_channels;
    uniform vec2 u_data_scale;
    uniform sampler2D u_channel_pos;
    uniform sampler2D u_cluster_color;

    varying vec4 v_color;
    varying vec2 v_box;

    // TODO: use VisPy transforms
    vec2 get_box_pos(vec2 box) {  // box = (cluster, channel)
        vec2 box_pos = texture2D(u_channel_pos,
                                 vec2(box.y / (n_channels - 1.), .5)).xy;
        box_pos = 2. * box_pos - 1.;
        box_pos.x += .1 * (box.x - .5 * (n_clusters - 1.));
        return box_pos;
    }

    vec3 get_color(float cluster) {
        return texture2D(u_cluster_color,
                         vec2(cluster / (n_clusters - 1.), .5)).xyz;
    }

    void main() {
        vec2 pos = u_data_scale * vec2(a_time, a_data.x);  // -1..1
        vec2 box_pos = get_box_pos(a_box);
        v_box = a_box;
        gl_Position = vec4($transform(pos + box_pos), 0., 1.);

        // Compute the waveform color as a function of the cluster color
        // and the mask.
        float mask = a_data.y;
        // TODO: store the colors in HSV in the texture?
        vec3 rgb = get_color(a_box.x);
        vec3 hsv = $rgb_to_hsv(rgb);
        // Change the saturation and value as a function of the mask.
        hsv.y = mask;
        hsv.z = .5 * (1. + mask);
        v_color.rgb = $hsv_to_rgb(hsv);
        v_color.a = .5;
    }
    """

    FRAG_SHADER = """
    varying vec4 v_color;
    varying vec2 v_box;

    void main() {
        if ((fract(v_box.x) > 0.) || (fract(v_box.y) > 0.))
            discard;
        gl_FragColor = v_color;
    }
    """

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def spike_clusters(self):
        """The clusters assigned to *all* spikes, not just the displayed
        spikes."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        self._spike_clusters = value

    @property
    def waveforms(self):
        """Displayed waveforms."""
        return self._waveforms

    @waveforms.setter
    def waveforms(self, value):
        assert isinstance(value, np.ndarray)
        # TODO: support sparse structures
        assert value.ndim == 3
        self.n_spikes, self.n_channels, self.n_samples = value.shape
        self._waveforms = value

    @property
    def masks(self):
        """Masks of the displayed waveforms."""
        return self._masks

    @masks.setter
    def masks(self, value):
        assert isinstance(value, np.ndarray)
        # TODO: support sparse structures
        assert value.ndim == 2
        assert value.shape == (self.n_spikes, self.n_channels)
        self._masks = value

    @property
    def spike_labels(self):
        """The list of spike labels to display, should correspond to the
        waveforms."""
        if self._spike_labels is None:
            self._spike_labels = np.arange(self.n_spikes).astype(np.int64)
        return self._spike_labels

    @spike_labels.setter
    def spike_labels(self, value):
        assert len(value) == self.n_spikes
        self._spike_labels = value

    @property
    def cluster_metadata(self):
        """A ClusterMetadata instance that holds information about all
        clusters."""
        return self._cluster_metadata

    @cluster_metadata.setter
    def cluster_metadata(self, value):
        self._cluster_metadata = value

    @property
    def channel_positions(self):
        """Array with the coordinates of all channels."""
        return self._channel_positions

    @channel_positions.setter
    def channel_positions(self, value):
        self._channel_positions = value

    @property
    def cluster_labels(self):
        """Clusters of the displayed spikes."""
        return _unique(self.spike_clusters[self.spike_labels])

    @property
    def n_clusters(self):
        return len(self.cluster_labels)

    @property
    def cluster_colors(self):
        """Colors of the displayed clusters."""
        clusters = self.cluster_labels
        return np.array([self._cluster_metadata[cluster]['color']
                         for cluster in clusters], dtype=np.float32)

    @property
    def box_scale(self):
        return self.program['u_data_scale']

    @box_scale.setter
    def box_scale(self, value):
        assert isinstance(value, tuple) and len(value) == 2
        self.program['u_data_scale'] = value
        self.update()

    # Data baking
    # -------------------------------------------------------------------------

    def bake_metadata(self):
        debug("bake metadata")
        u_cluster_color = self.cluster_colors.reshape((1, self.n_clusters, -1))
        self.program['u_cluster_color'] = Texture2D(u_cluster_color)

    def bake_channel_positions(self):
        debug("bake channel pos")
        # WARNING: channel_positions must be in [0,1] because we have a
        # texture.
        u_channel_pos = np.dstack((self.channel_positions.
                                  reshape((1, self.n_channels, 2)),
                                  np.zeros((1, self.n_channels, 1),
                                           dtype=np.float32)))
        self.program['u_channel_pos'] = Texture2D(u_channel_pos,
                                                  wrapping='clamp_to_edge')

    def bake_spikes(self):
        debug("bake spikes")

        # Bake masks.
        masks = np.repeat(self._masks.ravel(), self.n_samples)
        self.program['a_data'] = np.c_[self._waveforms.ravel(),
                                       masks.ravel()]

        # TODO: SparseCSR, this should just be 'channel'
        self._channels_per_spike = np.tile(np.arange(self.n_channels).
                                           astype(np.float32),
                                           self.n_spikes)

        # TODO: SparseCSR, this should be np.diff(spikes_ptr)
        self._n_channels_per_spike = self.n_channels * np.ones(self.n_spikes,
                                                               dtype=np.int32)

        self._n_waveforms = np.sum(self._n_channels_per_spike)

        # TODO: precompute this with a maximum number of waveforms?
        a_time = np.tile(np.linspace(-1., 1., self.n_samples).
                         astype(np.float32),
                         self._n_waveforms)

        self.program['a_time'] = a_time
        self.program['n_clusters'] = self.n_clusters
        self.program['n_channels'] = self.n_channels

    def bake_clusters(self):
        debug("bake clusters")
        a_cluster = np.repeat(self.spike_clusters[self.spike_labels],
                              self._n_channels_per_spike * self.n_samples)
        a_channel = np.repeat(self._channels_per_spike, self.n_samples)
        a_box = np.c_[a_cluster, a_channel].astype(np.float32)

        self.program['a_box'] = a_box

    def bake(self, changed=None):
        """Prepare and upload the data on the GPU."""
        if self.n_spikes is None:
            return
        if changed is None:
            changed = 'spikes'
        if _check_order(changed, 'spikes'):
            self.bake_spikes()
            self.bake_channel_positions()
        if _check_order(changed, 'clusters'):
            self.bake_clusters()
        if _check_order(changed, 'cluster_metadata'):
            self.bake_metadata()
        self._is_baked = True

    def __init__(self, **kwargs):
        super(Waveforms, self).__init__(**kwargs)
        self.n_spikes, self.n_channels, self.n_samples = None, None, None
        self._spike_clusters = None
        self._waveforms = None
        self._spike_labels = None
        self._is_baked = False

        self.program = ModularProgram(self.VERT_SHADER, self.FRAG_SHADER)
        self.program.vert['rgb_to_hsv'] = Function(RGB_TO_HSV)
        self.program.vert['hsv_to_rgb'] = Function(HSV_TO_RGB)
        self.program['u_data_scale'] = (.03, .02)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def draw(self, event):
        if not self._is_baked:
            self.bake()
        if self._is_baked:
            self.program.draw('line_strip')


class WaveformView(PanZoomCanvas):
    def __init__(self, **kwargs):
        super(WaveformView, self).__init__(**kwargs)
        self.visual = Waveforms()

    def on_key_press(self, event):
        # TODO: more interactivity
        # TODO: keyboard shortcut manager
        super(WaveformView, self).on_key_press(event)
        if event.key == '+':
            u, v = self.visual.box_scale
            self.visual.box_scale = (u, v*1.1)
        if event.key == '-':
            u, v = self.visual.box_scale
            self.visual.box_scale = (u, v/1.1)
