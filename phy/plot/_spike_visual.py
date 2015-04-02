# -*- coding: utf-8 -*-

"""Base spike visual."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo
from vispy.gloo import Texture2D
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram

from ._vispy_utils import _load_shader
from ..utils.array import _unique, _as_array
from ..utils.logging import debug


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class BaseSpikeVisual(Visual):
    _shader_name = ''
    _gl_draw_mode = ''

    def __init__(self, **kwargs):
        super(BaseSpikeVisual, self).__init__(**kwargs)
        self.n_spikes = None
        self._spike_clusters = None
        self._spike_ids = None
        self._to_bake = []

        vertex = _load_shader(self._shader_name + '.vert')
        fragment = _load_shader(self._shader_name + '.frag')

        self.program = ModularProgram(vertex, fragment)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    # Data properties
    # -------------------------------------------------------------------------

    def _set_or_assert_n_spikes(self, arr):
        """If n_spikes is None, set it using the array's shape. Otherwise,
        check that the array has n_spikes rows."""
        if self.n_spikes is None:
            self.n_spikes = arr.shape[0]
        assert arr.shape[0] == self.n_spikes

    def set_to_bake(self, *bakes):
        for bake in bakes:
            if bake not in self._to_bake:
                self._to_bake.append(bake)

    @property
    def spike_clusters(self):
        """The clusters assigned to *all* spikes, not just the displayed
        spikes."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        """Set all spike clusters."""
        value = _as_array(value)
        self._spike_clusters = value
        self.set_to_bake('spikes_clusters')

    @property
    def masks(self):
        """Masks of the displayed waveforms."""
        return self._masks

    @masks.setter
    def masks(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        # TODO: support sparse structures
        assert value.ndim == 2
        assert value.shape == (self.n_spikes, self.n_channels)
        self._masks = value
        self.set_to_bake('spikes')

    @property
    def spike_ids(self):
        """The list of spike ids to display, should correspond to the
        waveforms."""
        if self._spike_ids is None:
            self._spike_ids = np.arange(self.n_spikes).astype(np.int64)
        return self._spike_ids

    @spike_ids.setter
    def spike_ids(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        self._spike_ids = value
        self.set_to_bake('spikes')

    # TODO: channel_ids

    @property
    def cluster_ids(self):
        """Clusters of the displayed spikes."""
        return _unique(self.spike_clusters[self.spike_ids])

    @property
    def n_clusters(self):
        return len(self.cluster_ids)

    @property
    def cluster_colors(self):
        """Colors of the displayed clusters."""
        return self._cluster_colors

    @cluster_colors.setter
    def cluster_colors(self, value):
        self._cluster_colors = _as_array(value)
        assert len(self._cluster_colors) == self.n_clusters
        self.set_to_bake('color')

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_color(self):
        u_cluster_color = self.cluster_colors.reshape((1, self.n_clusters, -1))
        u_cluster_color = (u_cluster_color * 255).astype(np.uint8)
        # TODO: more efficient to update the data from an existing texture
        self.program['u_cluster_color'] = Texture2D(u_cluster_color)
        debug("bake color", u_cluster_color.shape)

    def _bake(self):
        """Prepare and upload the data on the GPU.

        Return whether something has been baked or not.

        """
        if self.n_spikes is None or self.n_spikes == 0:
            return
        n_bake = len(self._to_bake)
        # Bake what needs to be baked.
        # WARNING: the bake functions are called in alphabetical order.
        # Tweak the names if there are dependencies between the functions.
        for bake in sorted(self._to_bake):
            # Name of the private baking method.
            name = '_bake_{0:s}'.format(bake)
            if hasattr(self, name):
                getattr(self, name)()
        self._to_bake = []
        return n_bake > 0

    def draw(self, event):
        """Draw the waveforms."""
        # Bake what needs to be baked at this point.
        self._bake()
        if self.n_spikes is not None and self.n_spikes > 0:
            self.program.draw(self._gl_draw_mode)
