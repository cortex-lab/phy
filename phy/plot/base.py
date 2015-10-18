# -*- coding: utf-8 -*-

"""Plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np

from vispy import gloo, config

from phy.utils._types import _as_array
from phy.io.array import _unique

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class _BakeVisual(object):
    _shader_name = ''
    _gl_draw_mode = ''

    def __init__(self, **kwargs):
        super(_BakeVisual, self).__init__(**kwargs)
        self._to_bake = []
        self._empty = True

        curdir = op.dirname(op.realpath(__file__))
        config['include_path'] = [op.join(curdir, 'glsl')]

        vertex = _load_shader(self._shader_name + '.vert')
        fragment = _load_shader(self._shader_name + '.frag')
        self.program = gloo.Program(vertex, fragment)

    @property
    def empty(self):
        """Specify whether the visual is currently empty or not."""
        return self._empty

    @empty.setter
    def empty(self, value):
        """Specify whether the visual is currently empty or not."""
        self._empty = value

    def set_to_bake(self, *bakes):
        """Mark data items to be prepared for GPU."""
        for bake in bakes:
            if bake not in self._to_bake:
                self._to_bake.append(bake)

    def _bake(self):
        """Prepare and upload the data on the GPU.

        Return whether something has been baked or not.

        """
        if self._empty:
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

    def draw(self):
        """Draw the waveforms."""
        # Bake what needs to be baked at this point.
        self._bake()
        if not self._empty:
            self.program.draw(self._gl_draw_mode)

    def update(self):
        self.draw()


class BaseSpikeVisual(_BakeVisual):
    """Base class for a VisPy visual showing spike data.

    There is a notion of displayed spikes and displayed clusters.

    """

    _transparency = True

    def __init__(self, **kwargs):
        super(BaseSpikeVisual, self).__init__(**kwargs)
        self.n_spikes = None
        self._spike_clusters = None
        self._spike_ids = None
        self._cluster_ids = None
        self._cluster_order = None
        self._cluster_colors = None
        self._update_clusters_automatically = True

        if self._transparency:
            gloo.set_state(clear_color='black', blend=True,
                           blend_func=('src_alpha', 'one_minus_src_alpha'))

    # Data properties
    # -------------------------------------------------------------------------

    def _set_or_assert_n_spikes(self, arr):
        """If n_spikes is None, set it using the array's shape. Otherwise,
        check that the array has n_spikes rows."""
        # TODO: improve this
        if self.n_spikes is None:
            self.n_spikes = arr.shape[0]
        assert arr.shape[0] == self.n_spikes

    def _update_clusters(self):
        self._cluster_ids = _unique(self._spike_clusters)

    @property
    def spike_clusters(self):
        """The clusters assigned to the displayed spikes."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        """Set all spike clusters."""
        value = _as_array(value)
        self._spike_clusters = value
        if self._update_clusters_automatically:
            self._update_clusters()
        self.set_to_bake('spikes_clusters')

    @property
    def cluster_order(self):
        """List of selected clusters in display order."""
        if self._cluster_order is None:
            return self._cluster_ids
        else:
            return self._cluster_order

    @cluster_order.setter
    def cluster_order(self, value):
        value = _as_array(value)
        assert sorted(value.tolist()) == sorted(self._cluster_ids)
        self._cluster_order = value

    @property
    def masks(self):
        """Masks of the displayed spikes."""
        return self._masks

    @masks.setter
    def masks(self, value):
        assert isinstance(value, np.ndarray)
        value = _as_array(value)
        if value.ndim == 1:
            value = value[None, :]
        self._set_or_assert_n_spikes(value)
        # TODO: support sparse structures
        assert value.ndim == 2
        assert value.shape == (self.n_spikes, self.n_channels)
        self._masks = value
        self.set_to_bake('spikes')

    @property
    def spike_ids(self):
        """Spike ids to display."""
        if self._spike_ids is None:
            self._spike_ids = np.arange(self.n_spikes)
        return self._spike_ids

    @spike_ids.setter
    def spike_ids(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        self._spike_ids = value
        self.set_to_bake('spikes')

    @property
    def cluster_ids(self):
        """Cluster ids of the displayed spikes."""
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, value):
        """Clusters of the displayed spikes."""
        self._cluster_ids = _as_array(value)

    @property
    def n_clusters(self):
        """Number of displayed clusters."""
        if self._cluster_ids is None:
            return None
        else:
            return len(self._cluster_ids)

    @property
    def cluster_colors(self):
        """Colors of the displayed clusters.

        The first color is the color of the smallest cluster.

        """
        return self._cluster_colors

    @cluster_colors.setter
    def cluster_colors(self, value):
        self._cluster_colors = _as_array(value)
        assert len(self._cluster_colors) >= self.n_clusters
        self.set_to_bake('cluster_color')

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_cluster_color(self):
        if self.n_clusters == 0:
            u_cluster_color = np.zeros((0, 0, 3))
        else:
            u_cluster_color = self.cluster_colors.reshape((1,
                                                           self.n_clusters,
                                                           3))
        assert u_cluster_color.ndim == 3
        assert u_cluster_color.shape[2] == 3
        u_cluster_color = (u_cluster_color * 255).astype(np.uint8)
        self.program['u_cluster_color'] = gloo.Texture2D(u_cluster_color)
