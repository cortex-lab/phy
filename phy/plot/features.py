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
        self.n_channels, self.n_features = None, None

    # Data properties
    # -------------------------------------------------------------------------

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

    # TODO:
    # spike_times

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_spikes(self):
        # TODO
        n_rows = 3
        n_boxes = n_rows * n_rows
        n_points = n_boxes * self.n_spikes

        # index increases from top to bottom, left to right
        # same as matrix indices (i, j) starting at 0
        positions = []
        masks = []
        boxes = []

        for i in range(n_rows):
            for j in range(n_rows):
                index = n_rows * i + j

                # TODO: improve this
                positions.append(self._features[:,
                                 [i, j], 0].astype(np.float32))

                # TODO: choose the mask
                masks.append(self._masks[:, i].astype(np.float32))
                boxes.append(index * np.ones(self.n_spikes, dtype=np.float32))

        positions = np.vstack(positions)
        masks = np.hstack(masks)
        boxes = np.hstack(boxes)

        assert positions.shape == (n_points, 2)
        assert masks.shape == (n_points,)
        assert boxes.shape == (n_points,)

        self.program['a_position'] = positions.copy()
        self.program['a_mask'] = masks
        self.program['a_box'] = boxes

        self.program['n_clusters'] = self.n_clusters
        self.program['n_rows'] = n_rows
        self.program['u_size'] = 5.

        debug("bake spikes", positions.shape)

    def _bake_spikes_clusters(self):
        n_boxes = 9  # TODO

        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters[self.spike_ids]
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_ids)

        a_cluster = np.tile(spike_clusters_idx, n_boxes).astype(np.float32)
        self.program['a_cluster'] = a_cluster
        debug("bake spikes clusters", spike_clusters_idx.shape)


class FeatureView(PanZoomCanvas):
    def __init__(self, **kwargs):
        super(FeatureView, self).__init__(**kwargs)
        self.visual = Features()


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
