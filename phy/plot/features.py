# -*- coding: utf-8 -*-

"""Plotting features."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo

from ._vispy_utils import (BaseSpikeVisual,
                           BaseSpikeCanvas,
                           BoxVisual,
                           AxisVisual,
                           _enable_depth_mask,
                           )
from ._panzoom import PanZoomGrid
from ..ext.six import string_types
from ..utils.array import _as_array, _index_of
from ..utils.logging import debug


#------------------------------------------------------------------------------
# Features sisual
#------------------------------------------------------------------------------

class FeatureVisual(BaseSpikeVisual):

    _shader_name = 'features'
    _gl_draw_mode = 'points'

    """FeatureVisual visual."""
    def __init__(self, **kwargs):
        super(FeatureVisual, self).__init__(**kwargs)

        self._features = None
        self._spike_samples = None
        self._dimensions = []
        self.n_channels, self.n_features = None, None
        self.n_rows = None
        self.program['u_size'] = 3.

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def spike_samples(self):
        return self._spike_samples

    @spike_samples.setter
    def spike_samples(self, value):
        assert isinstance(value, np.ndarray)
        assert value.shape == (self.n_spikes,)
        self._spike_samples = value

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
        self._empty = self.n_spikes == 0
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
            t = self._spike_samples
            # Normalize time feature.
            m = t.max()
            if m > 0:
                t = -1. + 2 * t / m
            return t

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

                fet_j = self._get_feature_dim(dim_j)
                # For non-time dimensions, the diagonal shows
                # a different feature on y (same channel than x).
                if i == j and dim_i != 'time' and self.n_features >= 1:
                    channel, feature = dim_i
                    # Choose the other feature on y axis.
                    feature = 1 - feature
                    fet_i = self._features[:, channel, feature]
                else:
                    fet_i = self._get_feature_dim(dim_i)

                # NOTE: we switch here because we want to plot
                # dim_i (y) over dim_j (x) on box (i, j).
                pos = np.c_[fet_j, fet_i]
                positions.append(pos)

                # TODO: how to choose the mask?
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

        debug("bake spikes", positions.shape)

    def _bake_spikes_clusters(self):
        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters
        # We take the cluster order into account here.
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_order)
        a_cluster = np.tile(spike_clusters_idx,
                            self.n_boxes).astype(np.float32)
        self.program['a_cluster'] = a_cluster
        self.program['n_clusters'] = self.n_clusters
        debug("bake spikes clusters", spike_clusters_idx.shape)

    @property
    def marker_size(self):
        return float(self.program['u_size'])

    @marker_size.setter
    def marker_size(self, value):
        value = np.clip(value, .1, 100)
        self.program['u_size'] = float(value)
        self.update()


class FeatureView(BaseSpikeCanvas):
    """Display features.

    Interactivity
    -------------

    Marker size:

    * Keyboard : Control and '+' or '-'

    """
    _visual_class = FeatureVisual

    def __init__(self, **kwargs):
        super(FeatureView, self).__init__(**kwargs)
        self.boxes = BoxVisual()
        self.axes = AxisVisual()
        _enable_depth_mask()

    def _create_pan_zoom(self):
        if self.visual.n_rows:
            self._pz = PanZoomGrid(n_rows=self.visual.n_rows)
            self._pz.add(self.visual.program)
            self._pz.add(self.axes.program)
            self._pz.aspect = None
            self._pz.attach(self)

    def _set_pan_constraints(self):
        n = len(self.visual.dimensions)
        xmin = np.empty((n, n))
        xmax = np.empty((n, n))
        ymin = np.empty((n, n))
        ymax = np.empty((n, n))
        gpza = np.empty((n, n), dtype=np.str)
        gpza.fill('b')
        for arr in (xmin, xmax, ymin, ymax):
            arr.fill(np.nan)
        _index_set = False
        if self.visual.dimensions is not None:
            for i, dim in enumerate(self.visual.dimensions):
                if dim == 'time':
                    ymin[i, :] = -1.
                    xmin[:, i] = -1.
                    ymax[i, :] = +1.
                    xmax[:, i] = +1.
                    xmin[i, i] = -1.
                    xmax[i, i] = +1.
                    # Only update one axis for time dimensions during
                    # global zoom.
                    gpza[i, :] = 'x'
                    gpza[:, i] = 'y'
                else:
                    # Set the current index to the first non-time axis.
                    if not _index_set:
                        self._pz._index = (i, i)
                    _index_set = True
        self._pz._xmin = xmin
        self._pz._xmax = xmax
        self._pz._ymin = ymin
        self._pz._ymax = ymax
        self._pz.global_pan_zoom_axis = gpza

    @property
    def dimensions(self):
        """Dimensions."""
        return self.visual.dimensions

    @dimensions.setter
    def dimensions(self, value):
        # WARNING: dimensions should be changed here, in the Canvas,
        # and not in the visual. This is to make sure that the boxes are
        # updated as well.
        self.visual.dimensions = value
        self.boxes.n_rows = self.visual.n_rows
        self.axes.n_rows = self.visual.n_rows
        self._create_pan_zoom()
        self.axes.positions = (0, 0)
        self._set_pan_constraints()
        self.update()

    @property
    def marker_size(self):
        """Marker size."""
        return self.visual.marker_size

    @marker_size.setter
    def marker_size(self, value):
        self.visual.marker_size = value
        self.update()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.axes.draw()
        self.visual.draw()
        self.boxes.draw()

    def on_key_press(self, event):
        coeff = .25
        if 'Control' in event.modifiers:
            if event.key == '+':
                self.marker_size += coeff
            if event.key == '-':
                self.marker_size -= coeff
            self.update()
