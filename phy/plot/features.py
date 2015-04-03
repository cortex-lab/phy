# -*- coding: utf-8 -*-

"""Plotting features."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo
from vispy.gloo import Texture2D
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram, Function, Variable

from ._vispy_utils import PanZoomCanvas, _load_shader
from ..ext.six import string_types
from ..utils.array import _unique, _as_array, _index_of, _normalize
from ..utils.logging import debug
from ..utils._color import _random_color
from ._spike_visual import BaseSpikeVisual


#------------------------------------------------------------------------------
# Features visual
#------------------------------------------------------------------------------

class Features(BaseSpikeVisual):

    _shader_name = 'features'
    _gl_draw_mode = 'points'

    """Features visual."""
    def __init__(self, **kwargs):
        super(Features, self).__init__(**kwargs)

        self._features = None
        self._spike_times = None
        self._dimensions = []
        self.n_channels, self.n_features = None, None
        self.n_rows = None

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def spike_times(self):
        return self._spike_times

    @spike_times.setter
    def spike_times(self, value):
        assert isinstance(value, np.ndarray)
        assert value.shape == (self.n_spikes,)
        self._spike_times = value

    @property
    def features(self):
        """Displayed features."""
        return self._features

    @features.setter
    def features(self, value):
        # WARNING: when setting new data, features need to be set first.
        # n_spikes will be set as a function of features.
        value = _as_array(value)
        # TODO: support sparse structures
        assert value.ndim == 3
        self.n_spikes, self.n_channels, self.n_features = value.shape
        self._features = value
        self.set_to_bake('spikes', 'spikes_clusters', 'color')

    def _check_dimension(self, dim):
        if isinstance(dim, tuple):
            assert len(dim) == 2
            channel, feature = dim
            assert 0 <= channel < self.n_channels
            assert 0 <= feature < self.n_features
        elif isinstance(dim, string_types):
            assert dim == 'time'
        else:
            raise ValueError('{0} should be (channel, feature) '.format(dim) +
                             'or "time".')

    def _get_feature_dim(self, dim):
        if isinstance(dim, (tuple, list)):
            channel, feature = dim
            return self._features[:, channel, feature]
        elif dim == 'time':
            return self._spike_times

    def _get_mask_dim(self, dim):
        if isinstance(dim, (tuple, list)):
            channel, feature = dim
            return self._masks[:, channel]
        elif dim == 'time':
            return np.ones(self.n_spikes)

    @property
    def dimensions(self):
        """Dimensions."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self.n_rows = len(value)
        for dim in value:
            self._check_dimension(dim)
        self._dimensions = value
        self.set_to_bake('spikes', 'spikes_clusters', 'color')

    @property
    def n_boxes(self):
        return self.n_rows * self.n_rows

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_spikes(self):
        n_points = self.n_boxes * self.n_spikes

        # index increases from top to bottom, left to right
        # same as matrix indices (i, j) starting at 0
        positions = []
        masks = []
        boxes = []

        for i in range(self.n_rows):
            for j in range(self.n_rows):
                index = self.n_rows * i + j

                dim_i = self._dimensions[i]
                dim_j = self._dimensions[j]
                positions.append(np.c_[self._get_feature_dim(dim_i),
                                       self._get_feature_dim(dim_j)])

                # TODO: choose the mask
                mask = self._get_mask_dim(dim_i)
                masks.append(mask.astype(np.float32))
                boxes.append(index * np.ones(self.n_spikes, dtype=np.float32))

        positions = np.vstack(positions).astype(np.float32)
        masks = np.hstack(masks)
        boxes = np.hstack(boxes)

        assert positions.shape == (n_points, 2)
        assert masks.shape == (n_points,)
        assert boxes.shape == (n_points,)

        self.program['a_position'] = positions.copy()
        self.program['a_mask'] = masks
        self.program['a_box'] = boxes

        self.program['n_clusters'] = self.n_clusters
        self.program['n_rows'] = self.n_rows
        self.program['u_size'] = 5.

        debug("bake spikes", positions.shape)

    def _bake_spikes_clusters(self):
        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters[self.spike_ids]
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_ids)

        a_cluster = np.tile(spike_clusters_idx,
                            self.n_boxes).astype(np.float32)
        self.program['a_cluster'] = a_cluster
        debug("bake spikes clusters", spike_clusters_idx.shape)


class FeatureView(PanZoomCanvas):
    def __init__(self, **kwargs):
        super(FeatureView, self).__init__(**kwargs)
        self.visual = Features()
        self.zoom_center = 'origin'
        self.pan_scale = 1  # TODO: link that to the visual's n_rows


def add_feature_view(session, backend=None):
    """Add a feature view in a session.

    This function binds the session events to the created feature view.

    The caller needs to show the feature view explicitly.

    """
    if backend in ('pyqt4', None):
        kwargs = {'always_on_top': True}
    else:
        kwargs = {}
    view = FeatureView(**kwargs)

    @session.connect
    def on_open():
        if session.model is None:
            return
        view.visual.spike_clusters = session.clustering.spike_clusters
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
        view.visual.features = session.model.features[spikes]
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
