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
                           LassoVisual,
                           _enable_depth_mask,
                           )
from ._panzoom import PanZoomGrid
from ..ext.six import string_types
from ..utils._types import _as_array
from ..utils.array import _index_of


#------------------------------------------------------------------------------
# Features visual
#------------------------------------------------------------------------------

class BaseFeatureVisual(BaseSpikeVisual):
    """Display a grid of multidimensional features."""

    _shader_name = None
    _gl_draw_mode = 'points'

    def __init__(self, **kwargs):
        super(BaseFeatureVisual, self).__init__(**kwargs)

        self._features = None
        self._spike_samples = None
        self._dimensions = []
        self._diagonal_dimensions = []
        self.n_channels, self.n_features = None, None
        self.n_rows = None

        _enable_depth_mask()

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def spike_samples(self):
        """Time samples of the displayed spikes."""
        return self._spike_samples

    @spike_samples.setter
    def spike_samples(self, value):
        assert isinstance(value, np.ndarray)
        assert value.shape == (self.n_spikes,)
        self._spike_samples = value

    @property
    def features(self):
        """Displayed features.

        This is a `(n_spikes, n_features)` array.

        """
        return self._features

    @features.setter
    def features(self, value):
        self._set_features_to_bake(value)

    def _set_features_to_bake(self, value):
        # WARNING: when setting new data, features need to be set first.
        # n_spikes will be set as a function of features.
        value = _as_array(value)
        # TODO: support sparse structures
        assert value.ndim == 3
        self.n_spikes, self.n_channels, self.n_features = value.shape
        self._features = value
        self._empty = self.n_spikes == 0
        self.set_to_bake('spikes',)

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

    def _get_feature_dim(self, data, dim):
        if isinstance(dim, (tuple, list)):
            channel, feature = dim
            return data[:, channel, feature]
        elif dim == 'time':
            t = self._spike_samples
            # Normalize time feature.
            m = t.max()
            if m > 0:
                t = (-1. + 2 * t / m) * .8
            return t

    def project(self, data, box):
        """Project data to a subplot's two-dimensional subspace.

        Parameters
        ----------
        data : array
            The shape is `(n_points, n_channels, n_features)`.
        box : 2-tuple
            The `(row, col)` of the box.

        Notes
        -----

        The coordinate system is always the world coordinate system, i.e.
        `[-1, 1]`.

        """
        i, j = box
        dim_i = self._dimensions[i]
        dim_j = self._dimensions[j] if i != j else self._diagonal_dimensions[i]

        fet_i = self._get_feature_dim(self._features, dim_i)
        fet_j = self._get_feature_dim(self._features, dim_j)

        # NOTE: we switch here because we want to plot
        # dim_i (y) over dim_j (x) on box (i, j).
        return np.c_[fet_j, fet_i]

    @property
    def dimensions(self):
        """Displayed dimensions.

        This is a list of items which can be:

        * tuple `(channel_id, feature_idx)`
        * `'time'`

        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._set_dimensions_to_bake(value)
        self.diagonal_dimensions = self._default_diagonal(value)

    def _default_diagonal(self, dimensions):
        return [((dim[0], min(1 - dim[1], self.n_features - 1))
                 if dim != 'time' else 'time')
                for dim in dimensions]

    @property
    def diagonal_dimensions(self):
        """Displayed dimensions on the diagonal y axis.

        This is a list of items which can be:

        * tuple `(channel_id, feature_idx)`
        * `'time'`

        """
        return self._diagonal_dimensions

    @diagonal_dimensions.setter
    def diagonal_dimensions(self, value):
        assert len(value) == self.n_rows
        self._diagonal_dimensions = value
        self._set_dimensions_to_bake(self._dimensions)

    def _set_dimensions_to_bake(self, value):
        self.n_rows = len(value)
        for dim in value:
            self._check_dimension(dim)
        self._dimensions = value
        self.set_to_bake('spikes',)

    @property
    def n_boxes(self):
        """Number of boxes in the grid."""
        return self.n_rows * self.n_rows

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_spikes(self):
        n_points = self.n_boxes * self.n_spikes

        # index increases from top to bottom, left to right
        # same as matrix indices (i, j) starting at 0
        positions = []
        boxes = []

        for i in range(self.n_rows):
            for j in range(self.n_rows):
                pos = self.project(self._features, (i, j))
                positions.append(pos)
                index = self.n_rows * i + j
                boxes.append(index * np.ones(self.n_spikes, dtype=np.float32))

        positions = np.vstack(positions).astype(np.float32)
        boxes = np.hstack(boxes)

        assert positions.shape == (n_points, 2)
        assert boxes.shape == (n_points,)

        self.program['a_position'] = positions.copy()
        self.program['a_box'] = boxes
        self.program['n_rows'] = self.n_rows


class BackgroundFeatureVisual(BaseFeatureVisual):
    """Display a grid of multidimensional features in the background."""

    _shader_name = 'features_bg'
    _transparency = False


class FeatureVisual(BaseFeatureVisual):
    """Display a grid of multidimensional features."""

    _shader_name = 'features'

    def __init__(self, **kwargs):
        super(FeatureVisual, self).__init__(**kwargs)
        self.program['u_size'] = 3.

    # Data properties
    # -------------------------------------------------------------------------

    def _set_features_to_bake(self, value):
        super(FeatureVisual, self)._set_features_to_bake(value)
        self.set_to_bake('spikes', 'spikes_clusters', 'color')

    def _get_mask_dim(self, dim):
        if isinstance(dim, (tuple, list)):
            channel, feature = dim
            return self._masks[:, channel]
        elif dim == 'time':
            return np.ones(self.n_spikes)

    def _set_dimensions_to_bake(self, value):
        super(FeatureVisual, self)._set_dimensions_to_bake(value)
        self.set_to_bake('spikes', 'spikes_clusters', 'color')

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

                pos = self.project(self._features, (i, j))
                positions.append(pos)

                # TODO: how to choose the mask?
                dim_i = self._dimensions[i]
                mask = self._get_mask_dim(dim_i)
                masks.append(mask.astype(np.float32))

                index = self.n_rows * i + j
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

    def _bake_spikes_clusters(self):
        # Get the spike cluster indices (between 0 and n_clusters-1).
        spike_clusters_idx = self.spike_clusters
        # We take the cluster order into account here.
        spike_clusters_idx = _index_of(spike_clusters_idx, self.cluster_order)
        a_cluster = np.tile(spike_clusters_idx,
                            self.n_boxes).astype(np.float32)
        self.program['a_cluster'] = a_cluster
        self.program['n_clusters'] = self.n_clusters

    @property
    def marker_size(self):
        """Marker size in pixels."""
        return float(self.program['u_size'])

    @marker_size.setter
    def marker_size(self, value):
        value = np.clip(value, .1, 100)
        self.program['u_size'] = float(value)
        self.update()


class FeatureView(BaseSpikeCanvas):
    """A VisPy canvas displaying features."""
    _visual_class = FeatureVisual

    def _create_visuals(self):
        self.boxes = BoxVisual()
        self.axes = AxisVisual()
        self.background = BackgroundFeatureVisual()
        self.lasso = LassoVisual()
        super(FeatureView, self)._create_visuals()

    def _create_pan_zoom(self):
        self._pz = PanZoomGrid()
        self._pz.add(self.visual.program)
        self._pz.add(self.background.program)
        self._pz.add(self.lasso.program)
        self._pz.add(self.axes.program)
        self._pz.aspect = None
        self._pz.attach(self)

    def _set_pan_constraints(self, dimensions):
        n = len(dimensions)
        xmin = np.empty((n, n))
        xmax = np.empty((n, n))
        ymin = np.empty((n, n))
        ymax = np.empty((n, n))
        gpza = np.empty((n, n), dtype=np.str)
        gpza.fill('b')
        for arr in (xmin, xmax, ymin, ymax):
            arr.fill(np.nan)
        _index_set = False
        if dimensions is not None:
            for i, dim in enumerate(dimensions):
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
                    gpza[i, i] = 'n'
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
        return self.background.dimensions

    @dimensions.setter
    def dimensions(self, value):
        # WARNING: dimensions should be changed here, in the Canvas,
        # and not in the visual. This is to make sure that the boxes are
        # updated as well.
        self.visual.dimensions = value
        self.update_dimensions(value)

    @property
    def diagonal_dimensions(self):
        """Dimensions."""
        return self.background.diagonal_dimensions

    @diagonal_dimensions.setter
    def diagonal_dimensions(self, value):
        # WARNING: diagonal_dimensions should be changed here, in the Canvas,
        # and not in the visual. This is to make sure that the boxes are
        # updated as well.
        self.visual.diagonal_dimensions = value
        self.background.diagonal_dimensions = value
        self.update()

    def update_dimensions(self, dimensions):
        n_rows = len(dimensions)
        if self.background.features is not None:
            self.background.dimensions = dimensions
        self.boxes.n_rows = n_rows
        self.lasso.n_rows = n_rows
        self.axes.n_rows = n_rows
        self.axes.positions = (0, 0)
        self._pz.n_rows = n_rows
        self._set_pan_constraints(dimensions)
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
        """Draw the features in a grid view."""
        gloo.clear(color=True, depth=True)
        self.axes.draw()
        self.background.draw()
        self.visual.draw()
        self.lasso.draw()
        self.boxes.draw()

    keyboard_shortcuts = {
        'marker_size_increase': 'ctrl+[+]',
        'marker_size_decrease': 'ctrl+[-]',
        'add_lasso_point': 'ctrl+left click',
        'clear_lasso': 'ctrl+right click',
    }

    def on_mouse_press(self, e):
        ctrl = e.modifiers == ('Control',)
        if not ctrl:
            return
        if e.button == 1:
            n_rows = self.lasso.n_rows

            box = self._pz._get_box(e.pos)
            self.lasso.box = box

            position = self._pz._normalize(e.pos)
            x, y = position
            x *= n_rows
            y *= -n_rows
            pos = (x, y)
            # pos = self._pz._map_box((x, y), inverse=True)
            pos = self._pz._map_pan_zoom(pos, inverse=True)
            self.lasso.add(pos.ravel())
        elif e.button == 2:
            self.lasso.clear()
        self.update()

    def on_key_press(self, event):
        """Handle key press events."""
        coeff = .25
        if 'Control' in event.modifiers:
            if event.key == '+':
                self.marker_size += coeff
            if event.key == '-':
                self.marker_size -= coeff
