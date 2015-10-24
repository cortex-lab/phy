# -*- coding: utf-8 -*-

"""Common visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .base import BaseVisual
from .transform import Range, GPU
from .utils import _enable_depth_mask


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
        if data_bounds is None:
            m, M = pos.min(axis=0), pos.max(axis=0)
            data_bounds = [m[0], m[1], M[0], M[1]]
        assert len(data_bounds) == 4
        assert data_bounds[0] < data_bounds[2]
        assert data_bounds[1] < data_bounds[3]

        # Set the transformed position.
        pos_tr = self.apply_cpu_transforms(pos)
        pos_tr = np.asarray(pos_tr, dtype=np.float32)
        assert pos_tr.shape == (n, 2)

        # Set the depth.
        if depth is None:
            depth = np.zeros(n, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        assert depth.shape == (n,)

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
    gl_primitive_type = 'lines'

    def get_transforms(self):
        pass

    def set_data(self):
        pass


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
