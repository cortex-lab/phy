# -*- coding: utf-8 -*-

"""Common visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy.gloo import Texture2D

from .base import BaseVisual
from .transform import Range, NDC
from .utils import (_tesselate_histogram,
                    _get_texture,
                    _get_array,
                    _get_data_bounds,
                    _get_pos,
                    _get_index,
                    )
from phy.utils import Bunch


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

DEFAULT_COLOR = (0.03, 0.57, 0.98, .75)


#------------------------------------------------------------------------------
# Visuals
#------------------------------------------------------------------------------

class ScatterVisual(BaseVisual):
    _default_marker_size = 10.
    _default_marker = 'disc'
    _default_color = DEFAULT_COLOR
    _supported_markers = (
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

    def __init__(self, marker=None):
        super(ScatterVisual, self).__init__()
        self.n_points = None

        # Set the marker type.
        self.marker = marker or self._default_marker
        assert self.marker in self._supported_markers

        self.set_shader('scatter')
        self.fragment_shader = self.fragment_shader.replace('%MARKER',
                                                            self.marker)
        self.set_primitive_type('points')
        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def vertex_count(x=None, y=None, pos=None, **kwargs):
        return y.size if y is not None else len(pos)

    @staticmethod
    def validate(x=None,
                 y=None,
                 pos=None,
                 color=None,
                 size=None,
                 depth=None,
                 data_bounds=None,
                 ):
        if pos is None:
            x, y = _get_pos(x, y)
            pos = np.c_[x, y]
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        # Validate the data.
        color = _get_array(color, (n, 4), ScatterVisual._default_color)
        size = _get_array(size, (n, 1), ScatterVisual._default_marker_size)
        depth = _get_array(depth, (n, 1), 0)
        data_bounds = _get_data_bounds(data_bounds, pos)
        assert data_bounds.shape[0] == n

        return Bunch(pos=pos, color=color, size=size,
                     depth=depth, data_bounds=data_bounds)

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        self.data_range.from_bounds = data.data_bounds
        pos_tr = self.transforms.apply(data.pos)
        self.program['a_position'] = np.c_[pos_tr, data.depth]
        self.program['a_size'] = data.size
        self.program['a_color'] = data.color


def _as_list(arr):
    if isinstance(arr, np.ndarray):
        if arr.ndim == 1:
            return [arr]
        elif arr.ndim == 2:
            return list(arr)
    assert isinstance(arr, list)
    return arr


def _min(arr):
    return arr.min() if len(arr) else 0


def _max(arr):
    return arr.max() if len(arr) else 1


def _validate_line_coord(x, n, default):
    assert n >= 0
    if x is None:
        x = default
    if not hasattr(x, '__len__'):
        x = x * np.ones(n)
    x = np.asarray(x, dtype=np.float32)
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        x = x[:, None]
    assert x.shape == (n, 1)
    return x


def _get_length(*args):
    for arg in args:
        if hasattr(arg, '__len__'):
            return len(arg)
    return 1


class PlotVisual(BaseVisual):
    _default_color = DEFAULT_COLOR
    allow_list = ('x', 'y')

    def __init__(self):
        super(PlotVisual, self).__init__()

        self.set_shader('plot')
        self.set_primitive_type('line_strip')

        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def validate(x=None,
                 y=None,
                 color=None,
                 depth=None,
                 data_bounds=None,
                 ):

        assert y is not None
        y = _as_list(y)

        if x is None:
            x = [np.linspace(-1., 1., len(_)) for _ in y]
        x = _as_list(x)

        # Remove empty elements.
        assert len(x) == len(y)

        assert [len(_) for _ in x] == [len(_) for _ in y]

        n_signals = len(x)

        if data_bounds is None:
            xmin = [_min(_) for _ in x]
            ymin = [_min(_) for _ in y]
            xmax = [_max(_) for _ in x]
            ymax = [_max(_) for _ in y]
            data_bounds = np.c_[xmin, ymin, xmax, ymax]

        color = _get_array(color, (n_signals, 4), PlotVisual._default_color)
        assert color.shape == (n_signals, 4)

        depth = _get_array(depth, (n_signals, 1), 0)
        assert depth.shape == (n_signals, 1)

        data_bounds = _get_data_bounds(data_bounds, length=n_signals)
        data_bounds = data_bounds.astype(np.float32)
        assert data_bounds.shape == (n_signals, 4)

        return Bunch(x=x, y=y,
                     color=color, depth=depth,
                     data_bounds=data_bounds)

    @staticmethod
    def vertex_count(y=None, **kwargs):
        """Take the output of validate() as input."""
        return y.size if isinstance(y, np.ndarray) else sum(len(_) for _ in y)

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)

        assert isinstance(data.y, list)
        n_signals = len(data.y)
        n_samples = [len(_) for _ in data.y]
        n = sum(n_samples)
        x = np.concatenate(data.x) if len(data.x) else np.array([])
        y = np.concatenate(data.y) if len(data.y) else np.array([])

        # Generate the position array.
        pos = np.empty((n, 2), dtype=np.float32)
        pos[:, 0] = x.ravel()
        pos[:, 1] = y.ravel()
        assert pos.shape == (n, 2)

        # Generate signal index.
        signal_index = np.repeat(np.arange(n_signals), n_samples)
        signal_index = _get_array(signal_index, (n, 1)).astype(np.float32)
        assert signal_index.shape == (n, 1)

        # Transform the positions.
        data_bounds = np.repeat(data.data_bounds, n_samples, axis=0)
        self.data_range.from_bounds = data_bounds
        pos_tr = self.transforms.apply(pos)

        # Position and depth.
        depth = np.repeat(data.depth, n_samples, axis=0)
        self.program['a_position'] = np.c_[pos_tr, depth]

        self.program['a_signal_index'] = signal_index
        self.program['u_plot_colors'] = Texture2D(_get_texture(data.color,
                                                  PlotVisual._default_color,
                                                  n_signals,
                                                  [0, 1]))
        self.program['n_signals'] = n_signals


class HistogramVisual(BaseVisual):
    _default_color = DEFAULT_COLOR

    def __init__(self):
        super(HistogramVisual, self).__init__()

        self.set_shader('histogram')
        self.set_primitive_type('triangles')

        self.data_range = Range([0, 0, 1, 1])
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def validate(hist=None,
                 color=None,
                 ylim=None):
        assert hist is not None
        hist = np.asarray(hist, np.float32)
        if hist.ndim == 1:
            hist = hist[None, :]
        assert hist.ndim == 2
        n_hists, n_bins = hist.shape

        # Validate the data.
        color = _get_array(color, (n_hists, 4), HistogramVisual._default_color)

        # Validate ylim.
        if ylim is None:
            ylim = hist.max() if hist.size > 0 else 1.
        ylim = np.atleast_1d(ylim)
        if len(ylim) == 1:
            ylim = np.tile(ylim, n_hists)
        if ylim.ndim == 1:
            ylim = ylim[:, np.newaxis]
        assert ylim.shape == (n_hists, 1)

        return Bunch(hist=hist,
                     ylim=ylim,
                     color=color,
                     )

    @staticmethod
    def vertex_count(hist, **kwargs):
        hist = np.atleast_2d(hist)
        n_hists, n_bins = hist.shape
        return 6 * n_hists * n_bins

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        hist = data.hist

        n_hists, n_bins = hist.shape
        n = self.vertex_count(hist)

        # NOTE: this must be set *before* `apply_cpu_transforms` such
        # that the histogram is correctly normalized.
        data_bounds = np.c_[np.zeros((n_hists, 2)),
                            n_bins * np.ones((n_hists, 1)),
                            data.ylim]
        data_bounds = np.repeat(data_bounds, 6 * n_bins, axis=0)
        self.data_range.from_bounds = data_bounds

        # Set the transformed position.
        pos = np.vstack(_tesselate_histogram(row) for row in hist)
        pos = pos.astype(np.float32)
        pos_tr = self.transforms.apply(pos)
        assert pos_tr.shape == (n, 2)
        self.program['a_position'] = pos_tr

        # Generate the hist index.
        self.program['a_hist_index'] = _get_index(n_hists, n_bins * 6, n)

        # Hist colors.
        self.program['u_color'] = _get_texture(data.color,
                                               self._default_color,
                                               n_hists, [0, 1])
        self.program['n_hists'] = n_hists


class TextVisual(BaseVisual):
    def __init__(self):  # pragma: no cover
        # TODO: this text visual
        super(TextVisual, self).__init__()
        self.set_shader('text')
        self.set_primitive_type('points')

    def set_data(self):
        pass


class LineVisual(BaseVisual):
    """Lines.

    Note: currently, all lines shall have the same color.

    """
    _default_color = (.3, .3, .3, 1.)

    def __init__(self, color=None):
        super(LineVisual, self).__init__()
        self.set_shader('simple')
        self.set_primitive_type('lines')
        self.color = color or self._default_color

        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def validate(x0=None,
                 y0=None,
                 x1=None,
                 y1=None,
                 # color=None,
                 data_bounds=None,
                 ):

        # Get the number of lines.
        n_lines = _get_length(x0, y0, x1, y1)
        x0 = _validate_line_coord(x0, n_lines, -1)
        y0 = _validate_line_coord(y0, n_lines, -1)
        x1 = _validate_line_coord(x1, n_lines, +1)
        y1 = _validate_line_coord(y1, n_lines, +1)

        assert x0.shape == y0.shape == x1.shape == y1.shape == (n_lines, 1)

        # By default, we assume that the coordinates are in NDC.
        if data_bounds is None:
            data_bounds = NDC

        # NOTE: currently, we don't support custom colors. We could do it
        # by replacing the uniform by an attribute in the shaders.
        # color = _get_array(color, (4,), LineVisual._default_color)
        # assert len(color) == 4

        data_bounds = _get_data_bounds(data_bounds, length=n_lines)
        data_bounds = data_bounds.astype(np.float32)
        assert data_bounds.shape == (n_lines, 4)

        return Bunch(x0=x0,
                     y0=y0,
                     x1=x1,
                     y1=y1,
                     # color=color,
                     data_bounds=data_bounds,
                     )

    @staticmethod
    def vertex_count(x0=None, y0=None, x1=None, y1=None, **kwargs):
        """Take the output of validate() as input."""
        return 2 * _get_length(x0, y0, x1, y1)

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        pos = np.c_[data.x0, data.y0, data.x1, data.y1].astype(np.float32)
        assert pos.ndim == 2
        assert pos.shape[1] == 4
        assert pos.dtype == np.float32
        n_lines = pos.shape[0]
        n_vertices = 2 * n_lines
        pos = pos.reshape((-1, 2))

        # Transform the positions.
        data_bounds = np.repeat(data.data_bounds, 2, axis=0)
        self.data_range.from_bounds = data_bounds
        pos_tr = self.transforms.apply(pos)

        # Position.
        assert pos_tr.shape == (n_vertices, 2)
        self.program['a_position'] = pos_tr

        # Color.
        # self.program['u_color'] = data.color
        self.program['u_color'] = self.color
