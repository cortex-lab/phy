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
                           _wrap_vispy,
                           )
from ._panzoom import PanZoomGrid
from ..ext.six import string_types
from ..utils._types import _as_array, _is_integer
from ..utils.array import _index_of, _unique
from ..utils._color import _selected_clusters_colors


#------------------------------------------------------------------------------
# Features visual
#------------------------------------------------------------------------------

def _get_feature_dim(dim, features=None, extra_features=None):
    if isinstance(dim, (tuple, list)):
        channel, feature = dim
        return features[:, channel, feature]
    elif isinstance(dim, string_types) and dim in extra_features:
        x, m, M = extra_features[dim]
        m, M = (float(m), float(M))
        x = _as_array(x, np.float32)
        # Normalize extra feature.
        d = float(max(1., M - m))
        x = (-1. + 2 * (x - m) / d) * .8
        return x


class BaseFeatureVisual(BaseSpikeVisual):
    """Display a grid of multidimensional features."""

    _shader_name = None
    _gl_draw_mode = 'points'

    def __init__(self, **kwargs):
        super(BaseFeatureVisual, self).__init__(**kwargs)

        self._features = None
        # Mapping {feature_name: array} where the array must have n_spikes
        # element.
        self._extra_features = {}
        self._x_dim = np.empty((0, 0), dtype=object)
        self._y_dim = np.empty((0, 0), dtype=object)
        self.n_channels, self.n_features = None, None
        self.n_rows = None

        _enable_depth_mask()

    def add_extra_feature(self, name, array, array_min, array_max):
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        if self.n_spikes:
            if array.shape != (self.n_spikes,):
                msg = ("Unable to add the extra feature "
                       "`{}`: ".format(name) +
                       "there should be {} ".format(self.n_spikes) +
                       "elements in the specified vector, not "
                       "{}.".format(len(array)))
                raise ValueError(msg)

        self._extra_features[name] = (array, array_min, array_max)

    @property
    def extra_features(self):
        return self._extra_features

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
        if _is_integer(dim):
            dim = (dim, 0)
        if isinstance(dim, tuple):
            assert len(dim) == 2
            channel, feature = dim
            assert _is_integer(channel)
            assert _is_integer(feature)
            assert 0 <= channel < self.n_channels
            assert 0 <= feature < self.n_features
        elif isinstance(dim, string_types):
            assert dim in self._extra_features
        elif dim:
            raise ValueError('{0} should be (channel, feature) '.format(dim) +
                             'or one of the extra features.')

    def project(self, box, features=None, extra_features=None):
        """Project data to a subplot's two-dimensional subspace.

        Parameters
        ----------
        box : 2-tuple
            The `(row, col)` of the box.
        features : array
        extra_features : dict

        Notes
        -----

        The coordinate system is always the world coordinate system, i.e.
        `[-1, 1]`.

        """
        i, j = box
        dim_x = self._x_dim[i, j]
        dim_y = self._y_dim[i, j]

        fet_x = _get_feature_dim(dim_x,
                                 features=features,
                                 extra_features=extra_features,
                                 )
        fet_y = _get_feature_dim(dim_y,
                                 features=features,
                                 extra_features=extra_features,
                                 )
        return np.c_[fet_x, fet_y]

    @property
    def x_dim(self):
        """Dimensions in the x axis of all subplots.

        This is a matrix of items which can be:

        * tuple `(channel_id, feature_idx)`
        * an extra feature name (string)

        """
        return self._x_dim

    @x_dim.setter
    def x_dim(self, value):
        self._x_dim = value
        self._update_dimensions()

    @property
    def y_dim(self):
        """Dimensions in the y axis of all subplots.

        This is a matrix of items which can be:

        * tuple `(channel_id, feature_idx)`
        * an extra feature name (string)

        """
        return self._y_dim

    @y_dim.setter
    def y_dim(self, value):
        self._y_dim = value
        self._update_dimensions()

    def _update_dimensions(self):
        """Update the GPU data afte the dimensions have changed."""
        self._check_dimension_matrix(self._x_dim)
        self._check_dimension_matrix(self._y_dim)
        self.set_to_bake('spikes',)

    def _check_dimension_matrix(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=object)
        assert value.ndim == 2
        assert value.shape[0] == value.shape[1]
        assert value.dtype == object
        self.n_rows = len(value)
        for dim in value.flat:
            self._check_dimension(dim)

    def set_dimension(self, axis, box, dim):
        matrix = self._x_dim if axis == 'x' else self._y_dim
        matrix[box] = dim
        self._update_dimensions()

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
                pos = self.project((i, j),
                                   features=self._features,
                                   extra_features=self._extra_features,
                                   )
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
        else:
            return np.ones(self.n_spikes)

    def _update_dimensions(self):
        super(FeatureVisual, self)._update_dimensions()
        self.set_to_bake('spikes_clusters', 'color')

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
                pos = self.project((i, j),
                                   features=self._features,
                                   extra_features=self._extra_features,
                                   )
                positions.append(pos)

                # The mask depends on both the `x` and `y` coordinates.
                mask = np.maximum(self._get_mask_dim(self._x_dim[i, j]),
                                  self._get_mask_dim(self._y_dim[i, j]))
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


def _iter_dimensions(dimensions):
    if isinstance(dimensions, dict):
        for box, dim in dimensions.items():
            yield (box, dim)
    elif isinstance(dimensions, np.ndarray):
        n_rows, n_cols = dimensions.shape
        for i in range(n_rows):
            for j in range(n_rows):
                yield (i, j), dimensions[i, j]
    elif isinstance(dimensions, list):
        for i in range(len(dimensions)):
            l = dimensions[i]
            for j in range(len(l)):
                dim = l[j]
                yield (i, j), dim


class FeatureView(BaseSpikeCanvas):
    """A VisPy canvas displaying features."""
    _visual_class = FeatureVisual
    _events = ('enlarge',)

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

    def init_grid(self, n_rows):
        """Initialize the view with a given number of rows.

        Note
        ----

        This function *must* be called before setting the attributes.

        """
        assert n_rows >= 0

        x_dim = np.empty((n_rows, n_rows), dtype=object)
        y_dim = np.empty((n_rows, n_rows), dtype=object)
        x_dim.fill('time')
        y_dim.fill((0, 0))

        self.visual.n_rows = n_rows
        # NOTE: update the private variable because we don't want dimension
        # checking at this point nor do we want to prepare the GPU data.
        self.visual._x_dim = x_dim
        self.visual._y_dim = y_dim

        self.background.n_rows = n_rows
        self.background._x_dim = x_dim
        self.background._y_dim = y_dim

        self.boxes.n_rows = n_rows
        self.lasso.n_rows = n_rows
        self.axes.n_rows = n_rows
        self.axes.xs = [0]
        self.axes.ys = [0]
        self._pz.n_rows = n_rows

        xmin = np.empty((n_rows, n_rows))
        xmax = np.empty((n_rows, n_rows))
        ymin = np.empty((n_rows, n_rows))
        ymax = np.empty((n_rows, n_rows))
        for arr in (xmin, xmax, ymin, ymax):
            arr.fill(np.nan)
        self._pz._xmin = xmin
        self._pz._xmax = xmax
        self._pz._ymin = ymin
        self._pz._ymax = ymax

        for i in range(n_rows):
            for j in range(n_rows):
                self._pz._xmin[i, j] = -1.
                self._pz._xmax[i, j] = +1.

    @property
    def x_dim(self):
        return self.visual.x_dim

    @property
    def y_dim(self):
        return self.visual.y_dim

    def _set_dimension(self, axis, box, dim):
        self.background.set_dimension(axis, box, dim)
        self.visual.set_dimension(axis, box, dim)
        min = self._pz._xmin if axis == 'x' else self._pz._ymin
        max = self._pz._xmax if axis == 'x' else self._pz._ymax
        if dim == 'time':
            # NOTE: the private variables are the matrices.
            min[box] = -1.
            max[box] = +1.
        else:
            min[box] = None
            max[box] = None

    def set_dimensions(self, axis, dimensions):
        for box, dim in _iter_dimensions(dimensions):
            self._set_dimension(axis, box, dim)
        self. _update_dimensions()

    def smart_dimension(self,
                        axis,
                        box,
                        dim,
                        ):
        """Smartify a dimension selection by ensuring x != y."""
        if not isinstance(dim, tuple):
            return dim
        n_features = self.visual.n_features
        # Find current dimensions.
        mat = self.x_dim if axis == 'x' else self.y_dim
        mat_other = self.x_dim if axis == 'y' else self.y_dim
        prev_dim = mat[box]
        prev_dim_other = mat_other[box]
        # Select smart new dimension.
        if not isinstance(prev_dim, string_types):
            channel, feature = dim
            prev_channel, prev_feature = prev_dim
            # Scroll the feature if the channel is the same.
            if prev_channel == channel:
                feature = (prev_feature + 1) % n_features
            # Scroll the feature if it is the same than in the other axis.
            if (prev_dim_other != 'time' and
                    prev_dim_other == (channel, feature)):
                feature = (feature + 1) % n_features
            dim = (channel, feature)
        return dim

    def _update_dimensions(self):
        self.background._update_dimensions()
        self.visual._update_dimensions()

    def set_data(self,
                 features=None,
                 n_rows=1,
                 x_dimensions=None,
                 y_dimensions=None,
                 masks=None,
                 spike_clusters=None,
                 extra_features=None,
                 background_features=None,
                 colors=None,
                 ):
        if features is not None:
            assert isinstance(features, np.ndarray)
            if features.ndim == 2:
                features = features[..., None]
            assert features.ndim == 3
        else:
            features = self.visual.features
        n_spikes, n_channels, n_features = features.shape

        if spike_clusters is None:
            spike_clusters = np.zeros(n_spikes, dtype=np.int32)
        cluster_ids = _unique(spike_clusters)
        n_clusters = len(cluster_ids)

        if masks is None:
            masks = np.ones(features.shape[:2], dtype=np.float32)

        if colors is None:
            colors = _selected_clusters_colors(n_clusters)

        self.visual.features = features

        if background_features is not None:
            assert features.shape[1:] == background_features.shape[1:]
            self.background.features = background_features.astype(np.float32)
        else:
            self.background.n_channels = self.visual.n_channels
            self.background.n_features = self.visual.n_features

        if masks is not None:
            self.visual.masks = masks

        self.visual.spike_clusters = spike_clusters
        assert spike_clusters.shape == (n_spikes,)

        self.visual.cluster_colors = colors

        # Dimensions.
        self.init_grid(n_rows)
        if not extra_features:
            extra_features = {'time': (np.linspace(0., 1., n_spikes), 0., 1.)}
        for name, (array, m, M) in (extra_features or {}).items():
            self.add_extra_feature(name, array, m, M)
        self.set_dimensions('x', x_dimensions or {(0, 0): 'time'})
        self.set_dimensions('y', y_dimensions or {(0, 0): (0, 0)})

        self.update()

    def add_extra_feature(self, name, array, array_min, array_max,
                          array_bg=None):
        self.visual.add_extra_feature(name, array, array_min, array_max)
        # Note: the background array has a different number of spikes.
        if array_bg is None:
            array_bg = array
        self.background.add_extra_feature(name, array_bg,
                                          array_min, array_max)

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
        'marker_size_increase': 'alt+',
        'marker_size_decrease': 'alt-',
        'add_lasso_point': 'shift+left click',
        'clear_lasso': 'shift+right click',
    }

    def on_mouse_press(self, e):
        control = e.modifiers == ('Control',)
        shift = e.modifiers == ('Shift',)
        if shift:
            if e.button == 1:
                # Lasso.
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
        elif control:
            # Enlarge.
            box = self._pz._get_box(e.pos)
            self.emit('enlarge',
                      box=box,
                      x_dim=self.x_dim[box],
                      y_dim=self.y_dim[box],
                      )

    def on_key_press(self, event):
        """Handle key press events."""
        coeff = .25
        if 'Alt' in event.modifiers:
            if event.key == '+':
                self.marker_size += coeff
            if event.key == '-':
                self.marker_size -= coeff


#------------------------------------------------------------------------------
# Plotting functions
#------------------------------------------------------------------------------

@_wrap_vispy
def plot_features(features, **kwargs):
    """Plot features.

    Parameters
    ----------

    features : ndarray
        The features to plot. A `(n_spikes, n_channels, n_features)` array.
    spike_clusters : ndarray (optional)
        A `(n_spikes,)` int array with the spike clusters.
    masks : ndarray (optional)
        A `(n_spikes, n_channels)` float array with the spike masks.
    n_rows : int
        Number of rows (= number of columns) in the grid view.
    x_dimensions : list
        List of dimensions for the x axis.
    y_dimensions : list
        List of dimensions for the yœ axis.
    extra_features : dict
        A dictionary `{feature_name: array}` where `array` has
        `n_spikes` elements.
    background_features : ndarray
        The background features. A `(n_spikes, n_channels, n_features)` array.

    """
    c = FeatureView(keys='interactive')
    c.set_data(features, **kwargs)
    return c
