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
from vispy.visuals.glsl.color import HSV_TO_RGB, RGB_TO_HSV

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
    # grid

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_spikes(self):

        # Bake masks.
        features = self._features.astype(np.float32)
        masks = self._masks.astype(np.float32)

        position = features[:, :2, 0].copy()
        self.program['a_position'] = position  # TODO: choose dimension
        self.program['a_mask'] = masks[:, 0]  # TODO: choose the mask
        debug("bake spikes", position.shape)

        self.program['n_clusters'] = self.n_clusters
        self.program['u_size'] = 10.

    def _bake_spikes_clusters(self):
        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters[self.spike_ids]
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_ids)

        a_box = np.zeros((self.n_spikes, 3), dtype=np.float32)
        a_box[:, 0] = spike_clusters_idx
        # TODO: row, col in a_box[1:]

        self.program['a_box'] = a_box
        debug("bake spikes clusters", a_box.shape)


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
