# -*- coding: utf-8 -*-

"""Common visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy.gloo import Texture2D

from .base import BaseVisual
from .transform import Range, GPU
from .utils import _enable_depth_mask


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _check_data_bounds(data_bounds):
    assert len(data_bounds) == 4
    assert data_bounds[0] < data_bounds[2]
    assert data_bounds[1] < data_bounds[3]


def _get_data_bounds(data_bounds, pos):
    if not len(pos):
        return data_bounds or [-1, -1, 1, 1]
    if data_bounds is None:
        m, M = pos.min(axis=0), pos.max(axis=0)
        data_bounds = [m[0], m[1], M[0], M[1]]
    _check_data_bounds(data_bounds)
    return data_bounds


#------------------------------------------------------------------------------
# Visuals
#------------------------------------------------------------------------------

class ScatterVisual(BaseVisual):
    shader_name = 'scatter'
    gl_primitive_type = 'points'
    _default_marker_size = 10.
    _supported_marker_types = (
        'arrow',
        'asterisk',
        'chevron',
        'clover',
        'club',
        'cross',
        'diamond',
        'disc',
        'ellipse',
        'hbar',
        'heart',
        'infinity',
        'pin',
        'ring',
        'spade',
        'square',
        'tag',
        'triangle',
        'vbar',
    )

    def __init__(self, marker_type=None):
        super(ScatterVisual, self).__init__()
        # Default bounds.
        self.data_bounds = [-1, -1, 1, 1]
        self.n_points = None

        # Set the marker type.
        self.marker_type = marker_type or 'disc'
        assert self.marker_type in self._supported_marker_types

        # Enable transparency.
        _enable_depth_mask()

    def get_shaders(self):
        v, f = super(ScatterVisual, self).get_shaders()
        # Replace the marker type in the shader.
        f = f.replace('%MARKER_TYPE', self.marker_type)
        return v, f

    def get_transforms(self):
        return [Range(from_bounds=self.data_bounds), GPU()]

    def set_data(self,
                 pos=None,
                 depth=None,
                 colors=None,
                 marker_type=None,
                 size=None,
                 data_bounds=None,
                 ):
        assert pos is not None
        pos = np.asarray(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        # Set the data bounds from the data.
        self.data_bounds = _get_data_bounds(data_bounds, pos)

        # Set the transformed position.
        pos_tr = self.apply_cpu_transforms(pos)
        pos_tr = np.asarray(pos_tr, dtype=np.float32)
        assert pos_tr.shape == (n, 2)

        # Set the depth.
        if depth is None:
            depth = np.zeros(n, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        assert depth.shape == (n,)

        # Set the a_position attribute.
        pos_depth = np.empty((n, 3), dtype=np.float32)
        pos_depth[:, :2] = pos_tr
        pos_depth[:, 2] = depth
        self.program['a_position'] = pos_depth

        # Set the marker size.
        if size is None:
            size = self._default_marker_size * np.ones(n, dtype=np.float32)
        size = np.asarray(size, dtype=np.float32)
        self.program['a_size'] = size

        # Set the group colors.
        if colors is None:
            colors = np.ones((n, 4), dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        assert colors.shape == (n, 4)
        self.program['a_color'] = colors


class PlotVisual(BaseVisual):
    shader_name = 'plot'
    gl_primitive_type = 'line_strip'

    def __init__(self):
        super(PlotVisual, self).__init__()
        self.data_bounds = [-1, -1, 1, 1]
        _enable_depth_mask()

    def get_transforms(self):
        return [Range(from_bounds=self.data_bounds),
                GPU(),
                Range(from_bounds=(-1, -1, 1, 1),
                      to_bounds='signal_bounds'),
                ]

    def set_data(self,
                 data=None,
                 depth=None,
                 data_bounds=None,
                 signal_bounds=None,
                 signal_colors=None,
                 ):
        assert data is not None
        data = np.asarray(data)
        assert data.ndim == 2
        n_signals, n_samples = data.shape
        n = n_signals * n_samples

        # Generate the x coordinates.
        x = np.linspace(-1., 1., n_samples)
        x = np.tile(x, n_signals)
        assert x.shape == (n,)

        # Generate the signal index.
        signal_index = np.arange(n_signals)
        signal_index = np.repeat(signal_index, n_samples)
        signal_index = signal_index.astype(np.float32)

        # Generate the (n, 2) pos array.
        pos = np.empty((n, 2), dtype=np.float32)
        pos[:, 0] = x
        pos[:, 1] = data.ravel()

        # Generate the complete data_bounds 4-tuple from the specified 2-tuple.
        if data_bounds is None:
            data_bounds = [data.min(), data.max()] if data.size else [-1, 1]
        assert len(data_bounds) == 2
        # Ensure that the data bounds are not degenerate.
        if data_bounds[0] == data_bounds[1]:
            data_bounds = [data_bounds[0] - 1, data_bounds[0] + 1]
        ymin, ymax = data_bounds
        data_bounds = [-1, ymin, 1, ymax]
        _check_data_bounds(data_bounds)
        self.data_bounds = data_bounds

        # Set the transformed position.
        pos_tr = self.apply_cpu_transforms(pos)
        pos_tr = np.asarray(pos_tr, dtype=np.float32)
        assert pos_tr.shape == (n, 2)

        # Set the depth.
        if depth is None:
            depth = np.zeros(n, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        assert depth.shape == (n,)

        # Set the a_position attribute.
        pos_depth = np.empty((n, 3), dtype=np.float32)
        pos_depth[:, :2] = pos_tr
        pos_depth[:, 2] = depth
        self.program['a_position'] = pos_depth

        # Signal index.
        self.program['a_signal_index'] = signal_index

        # Signal bounds (positions).
        if signal_bounds is None:
            signal_bounds = np.tile([-1, -1, 1, 1], (n_signals, 1))
        assert signal_bounds.shape == (n_signals, 4)
        # Convert to 3D texture.
        signal_bounds = signal_bounds[np.newaxis, ...].astype(np.float32)
        assert signal_bounds.shape == (1, n_signals, 4)
        # NOTE: we need to cast the texture to [0, 255] (uint8).
        # This is easy as soon as we assume that the signal bounds are in
        # [-1, 1].
        assert np.all(signal_bounds >= -1)
        assert np.all(signal_bounds <= 1)
        signal_bounds = 127 * (signal_bounds + 1)
        assert np.all(signal_bounds >= 0)
        assert np.all(signal_bounds <= 255)
        signal_bounds = signal_bounds.astype(np.uint8)
        self.program['u_signal_bounds'] = Texture2D(signal_bounds)

        # Signal colors.
        if signal_colors is None:
            signal_colors = np.ones((n_signals, 4), dtype=np.float32)
        assert signal_colors.shape == (n_signals, 4)
        # Convert to 3D texture.
        signal_colors = signal_colors[np.newaxis, ...].astype(np.float32)
        assert signal_colors.shape == (1, n_signals, 4)
        self.program['u_signal_colors'] = Texture2D(signal_colors)

        # Number of signals.
        self.program['n_signals'] = n_signals


class HistogramVisual(BaseVisual):
    shader_name = 'plot'
    gl_primitive_type = 'triangles'

    def get_transforms(self):
        pass

    def set_data(self):
        pass


class TextVisual(BaseVisual):
    shader_name = 'text'
    gl_primitive_type = 'points'

    def get_transforms(self):
        pass

    def set_data(self):
        pass
