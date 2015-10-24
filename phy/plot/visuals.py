# -*- coding: utf-8 -*-

"""Common visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy.gloo import Texture2D

from .base import BaseVisual
from .transform import Range, GPU
from .utils import _enable_depth_mask, _tesselate_histogram


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
        self.program['a_signal_index'] = signal_index

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
    shader_name = 'histogram'
    gl_primitive_type = 'triangles'

    def __init__(self):
        super(HistogramVisual, self).__init__()
        self.n_bins = 0
        self.hist_max = 1

    def get_transforms(self):
        return [Range(from_bounds=[0, 0, self.n_bins, self.hist_max],
                      to_bounds=[0, 0, 1, 1]),
                GPU(),
                Range(from_bounds='hist_bounds',   # (0, 0, 1, v)
                      to_bounds=(-1, -1, 1, 1)),
                ]

    def set_data(self,
                 hist=None,
                 hist_lims=None,
                 hist_colors=None,
                 ):
        assert hist is not None
        hist = np.atleast_2d(hist)
        assert hist.ndim == 2
        n_hists, n_bins = hist.shape
        n = 6 * n_hists * n_bins
        self.n_bins = n_bins

        # Generate hist_max.
        hist_max = hist.max() if hist.size else 1.
        hist_max = float(hist_max)
        hist_max = hist_max if hist_max > 0 else 1.
        assert hist_max > 0
        self.hist_max = hist_max

        # Concatenate all histograms.
        pos = np.vstack(_tesselate_histogram(row) for row in hist)
        assert pos.shape == (n, 2)

        # Set the transformed position.
        pos_tr = self.apply_cpu_transforms(pos)
        pos_tr = np.asarray(pos_tr, dtype=np.float32)
        assert pos_tr.shape == (n, 2)
        self.program['a_position'] = pos_tr

        # Generate the hist index.
        hist_index = np.arange(n_hists)
        # 6 * n_bins vertices per histogram.
        hist_index = np.repeat(hist_index, n_bins * 6)
        hist_index = hist_index.astype(np.float32)
        assert hist_index.shape == (n,)
        self.program['a_hist_index'] = hist_index

        # Hist colors.
        if hist_colors is None:
            hist_colors = np.ones((n_hists, 4), dtype=np.float32)
        assert hist_colors.shape == (n_hists, 4)
        # Convert to 3D texture.
        hist_colors = hist_colors[np.newaxis, ...].astype(np.float32)
        assert hist_colors.shape == (1, n_hists, 4)
        self.program['u_hist_colors'] = Texture2D(hist_colors)

        # Hist bounds.
        if hist_lims is None:
            hist_lims = hist_max * np.ones(n_hists)
        hist_lims = np.asarray(hist_lims, dtype=np.float32)
        # NOTE: hist_lims is now relative to hist_max (what is on the GPU).
        hist_lims = hist_lims / hist_max
        assert hist_lims.shape == (n_hists,)
        # Now, we create the 4-tuples for the bounds: [0, 0, 1, hists_lim].
        hist_bounds = np.zeros((n_hists, 4), dtype=np.float32)
        hist_bounds[:, 2] = 1
        hist_bounds[:, 3] = hist_lims
        # Convert to 3D texture.
        hist_bounds = hist_bounds[np.newaxis, ...].astype(np.float32)
        assert hist_bounds.shape == (1, n_hists, 4)
        assert np.all(hist_bounds >= 0)
        assert np.all(hist_bounds <= 10)
        # NOTE: necessary because VisPy silently clips textures to [0, 1].
        hist_bounds /= 10.
        self.program['u_hist_bounds'] = Texture2D(hist_bounds)

        self.program['n_hists'] = n_hists


class TextVisual(BaseVisual):
    shader_name = 'text'
    gl_primitive_type = 'points'

    def get_transforms(self):
        pass

    def set_data(self):
        pass
