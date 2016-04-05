# -*- coding: utf-8 -*-

"""Common visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import gzip
import os.path as op

import numpy as np
from six import string_types
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
# Scatter visuals
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
                 data_bounds='auto',
                 ):
        if pos is None:
            x, y = _get_pos(x, y)
            pos = np.c_[x, y]
        pos = np.asarray(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        # Validate the data.
        color = _get_array(color, (n, 4),
                           ScatterVisual._default_color,
                           dtype=np.float32)
        size = _get_array(size, (n, 1), ScatterVisual._default_marker_size)
        depth = _get_array(depth, (n, 1), 0)
        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, pos)
            assert data_bounds.shape[0] == n

        return Bunch(pos=pos, color=color, size=size,
                     depth=depth, data_bounds=data_bounds)

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        if data.data_bounds is not None:
            self.data_range.from_bounds = data.data_bounds
            pos_tr = self.transforms.apply(data.pos)
        else:
            pos_tr = data.pos
        pos_tr = np.c_[pos_tr, data.depth]
        self.program['a_position'] = pos_tr.astype(np.float32)
        self.program['a_size'] = data.size.astype(np.float32)
        self.program['a_color'] = data.color.astype(np.float32)


class UniformScatterVisual(BaseVisual):
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

    def __init__(self, marker=None, color=None, size=None):
        super(UniformScatterVisual, self).__init__()

        # Set the marker type.
        self.marker = marker or self._default_marker
        assert self.marker in self._supported_markers

        self.set_shader('uni_scatter')
        self.fragment_shader = self.fragment_shader.replace('%MARKER',
                                                            self.marker)

        self.color = color or self._default_color
        self.marker_size = size or self._default_marker_size

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
                 masks=None,
                 data_bounds='auto',
                 ):
        if pos is None:
            x, y = _get_pos(x, y)
            pos = np.c_[x, y]
        pos = np.asarray(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        masks = _get_array(masks, (n, 1), 1., np.float32)
        assert masks.shape == (n, 1)

        # Validate the data.
        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, pos)
            assert data_bounds.shape[0] == n

        return Bunch(pos=pos,
                     masks=masks,
                     data_bounds=data_bounds,
                     )

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        if data.data_bounds is not None:
            self.data_range.from_bounds = data.data_bounds
            pos_tr = self.transforms.apply(data.pos)
        else:
            pos_tr = data.pos

        masks = data.masks

        self.program['a_position'] = pos_tr.astype(np.float32)
        self.program['a_mask'] = masks.astype(np.float32)

        self.program['u_size'] = self.marker_size
        self.program['u_color'] = self.color
        self.program['u_mask_max'] = _max(masks)


#------------------------------------------------------------------------------
# Plot visuals
#------------------------------------------------------------------------------

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
                 data_bounds='auto',
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

        if isinstance(data_bounds, string_types) and data_bounds == 'auto':
            xmin = [_min(_) for _ in x]
            ymin = [_min(_) for _ in y]
            xmax = [_max(_) for _ in x]
            ymax = [_max(_) for _ in y]
            data_bounds = np.c_[xmin, ymin, xmax, ymax]

        color = _get_array(color, (n_signals, 4),
                           PlotVisual._default_color,
                           dtype=np.float32,
                           )
        assert color.shape == (n_signals, 4)

        depth = _get_array(depth, (n_signals, 1), 0)
        assert depth.shape == (n_signals, 1)

        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, length=n_signals)
            data_bounds = data_bounds.astype(np.float64)
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
        pos = np.empty((n, 2), dtype=np.float64)
        pos[:, 0] = x.ravel()
        pos[:, 1] = y.ravel()
        assert pos.shape == (n, 2)

        # Generate the color attribute.
        color = data.color
        assert color.shape == (n_signals, 4)
        color = np.repeat(color, n_samples, axis=0)
        assert color.shape == (n, 4)

        # Generate signal index.
        signal_index = np.repeat(np.arange(n_signals), n_samples)
        signal_index = _get_array(signal_index, (n, 1))
        assert signal_index.shape == (n, 1)

        # Transform the positions.
        if data.data_bounds is not None:
            data_bounds = np.repeat(data.data_bounds, n_samples, axis=0)
            self.data_range.from_bounds = data_bounds
            pos = self.transforms.apply(pos)

        # Position and depth.
        depth = np.repeat(data.depth, n_samples, axis=0)
        self.program['a_position'] = np.c_[pos, depth].astype(np.float32)
        self.program['a_color'] = color.astype(np.float32)
        self.program['a_signal_index'] = signal_index.astype(np.float32)


class UniformPlotVisual(BaseVisual):
    _default_color = DEFAULT_COLOR
    allow_list = ('x', 'y')

    def __init__(self, color=None, depth=None):
        super(UniformPlotVisual, self).__init__()

        self.set_shader('uni_plot')
        self.set_primitive_type('line_strip')
        self.color = color or self._default_color

        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def validate(x=None,
                 y=None,
                 masks=None,
                 data_bounds='auto',
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

        masks = _get_array(masks, (n_signals, 1), 1., np.float32)
        assert masks.shape == (n_signals, 1)

        if isinstance(data_bounds, string_types) and data_bounds == 'auto':
            xmin = [_min(_) for _ in x]
            ymin = [_min(_) for _ in y]
            xmax = [_max(_) for _ in x]
            ymax = [_max(_) for _ in y]
            data_bounds = np.c_[xmin, ymin, xmax, ymax]

        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, length=n_signals)
            data_bounds = data_bounds.astype(np.float64)
            assert data_bounds.shape == (n_signals, 4)

        return Bunch(x=x, y=y, masks=masks,
                     data_bounds=data_bounds,
                     )

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
        pos = np.empty((n, 2), dtype=np.float64)
        pos[:, 0] = x.ravel()
        pos[:, 1] = y.ravel()
        assert pos.shape == (n, 2)

        # Generate signal index.
        signal_index = np.repeat(np.arange(n_signals), n_samples)
        signal_index = _get_array(signal_index, (n, 1))
        assert signal_index.shape == (n, 1)

        # Masks.
        masks = np.repeat(data.masks, n_samples, axis=0)

        # Transform the positions.
        if data.data_bounds is not None:
            data_bounds = np.repeat(data.data_bounds, n_samples, axis=0)
            self.data_range.from_bounds = data_bounds
            pos = self.transforms.apply(pos)

        assert pos.shape == (n, 2)
        assert signal_index.shape == (n, 1)
        assert masks.shape == (n, 1)

        # Position and depth.
        self.program['a_position'] = pos.astype(np.float32)
        self.program['a_signal_index'] = signal_index.astype(np.float32)
        self.program['a_mask'] = masks.astype(np.float32)

        self.program['u_color'] = self.color
        self.program['u_mask_max'] = _max(masks)


#------------------------------------------------------------------------------
# Other visuals
#------------------------------------------------------------------------------

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
        hist = np.asarray(hist, np.float64)
        if hist.ndim == 1:
            hist = hist[None, :]
        assert hist.ndim == 2
        n_hists, n_bins = hist.shape

        # Validate the data.
        color = _get_array(color, (n_hists, 4),
                           HistogramVisual._default_color,
                           dtype=np.float32,
                           )

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
        pos_tr = self.transforms.apply(pos)
        assert pos_tr.shape == (n, 2)
        self.program['a_position'] = pos_tr.astype(np.float32)

        # Generate the hist index.
        hist_index = _get_index(n_hists, n_bins * 6, n)
        self.program['a_hist_index'] = hist_index.astype(np.float32)

        # Hist colors.
        tex = _get_texture(data.color, self._default_color, n_hists, [0, 1])
        self.program['u_color'] = tex.astype(np.float32)
        self.program['n_hists'] = n_hists


class TextVisual(BaseVisual):
    """Display strings at multiple locations.

    Currently, the color, font family, and font size is not customizable.

    """
    _default_color = (1., 1., 1., 1.)

    def __init__(self, color=None):
        super(TextVisual, self).__init__()
        self.set_shader('text')
        self.set_primitive_type('triangles')
        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

        # Load the font.
        curdir = op.realpath(op.dirname(__file__))
        font_name = 'SourceCodePro-Regular'
        font_size = 32
        # The font texture is gzipped.
        fn = '%s-%d.npy.gz' % (font_name, font_size)
        with gzip.open(op.join(curdir, 'static', fn), 'rb') as f:
            self._tex = np.load(f)
        with open(op.join(curdir, 'static', 'chars.txt'), 'r') as f:
            self._chars = f.read()
        self.color = color if color is not None else self._default_color
        assert len(self.color) == 4

    def _get_glyph_indices(self, s):
        return [self._chars.index(char) for char in s]

    @staticmethod
    def validate(pos=None, text=None, anchor=None,
                 data_bounds='auto',
                 ):

        if text is None:
            text = []
        if isinstance(text, string_types):
            text = [text]
        if pos is None:
            pos = np.zeros((len(text), 2))

        assert pos is not None
        pos = np.atleast_2d(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n_text = pos.shape[0]
        assert len(text) == n_text

        anchor = anchor if anchor is not None else (0., 0.)
        anchor = np.atleast_2d(anchor)
        if anchor.shape[0] == 1:
            anchor = np.repeat(anchor, n_text, axis=0)
        assert anchor.ndim == 2
        assert anchor.shape == (n_text, 2)

        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, pos)
            assert data_bounds.shape[0] == n_text
            data_bounds = data_bounds.astype(np.float64)
            assert data_bounds.shape == (n_text, 4)

        return Bunch(pos=pos, text=text, anchor=anchor,
                     data_bounds=data_bounds)

    @staticmethod
    def vertex_count(pos=None, **kwargs):
        """Take the output of validate() as input."""
        # Total number of glyphs * 6 (6 vertices per glyph).
        return sum(map(len, kwargs['text'])) * 6

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        pos = data.pos.astype(np.float64)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        assert pos.dtype == np.float64

        # Concatenate all strings.
        text = data.text
        lengths = list(map(len, text))
        text = ''.join(text)
        a_char_index = self._get_glyph_indices(text)
        n_glyphs = len(a_char_index)

        tex = self._tex
        glyph_height = tex.shape[0] // 6
        glyph_width = tex.shape[1] // 16
        glyph_size = (glyph_width, glyph_height)

        # Position of all glyphs.
        a_position = np.repeat(pos, lengths, axis=0)
        if not len(lengths):
            a_glyph_index = np.zeros((0,))
        else:
            a_glyph_index = np.concatenate([np.arange(n) for n in lengths])
        a_quad_index = np.arange(6)

        a_anchor = data.anchor

        a_position = np.repeat(a_position, 6, axis=0)
        a_glyph_index = np.repeat(a_glyph_index, 6)
        a_quad_index = np.tile(a_quad_index, n_glyphs)
        a_char_index = np.repeat(a_char_index, 6)

        a_anchor = np.repeat(a_anchor, lengths, axis=0)
        a_anchor = np.repeat(a_anchor, 6, axis=0)

        a_lengths = np.repeat(lengths, lengths)
        a_lengths = np.repeat(a_lengths, 6)

        n_vertices = n_glyphs * 6

        # Transform the positions.
        if data.data_bounds is not None:
            data_bounds = data.data_bounds
            data_bounds = np.repeat(data_bounds, lengths, axis=0)
            data_bounds = np.repeat(data_bounds, 6, axis=0)
            assert data_bounds.shape == (n_vertices, 4)
            self.data_range.from_bounds = data_bounds
            pos_tr = self.transforms.apply(a_position)
            assert pos_tr.shape == (n_vertices, 2)
        else:
            pos_tr = a_position

        assert pos_tr.shape == (n_vertices, 2)
        assert a_glyph_index.shape == (n_vertices,)
        assert a_quad_index.shape == (n_vertices,)
        assert a_char_index.shape == (n_vertices,)
        assert a_anchor.shape == (n_vertices, 2)
        assert a_lengths.shape == (n_vertices,)

        self.program['a_position'] = pos_tr.astype(np.float32)
        self.program['a_glyph_index'] = a_glyph_index.astype(np.float32)
        self.program['a_quad_index'] = a_quad_index.astype(np.float32)
        self.program['a_char_index'] = a_char_index.astype(np.float32)
        self.program['a_anchor'] = a_anchor.astype(np.float32)
        self.program['a_lengths'] = a_lengths.astype(np.float32)

        self.program['u_glyph_size'] = glyph_size
        # TODO: color

        self.program['u_tex'] = Texture2D(tex[::-1, :])


class LineVisual(BaseVisual):
    """Lines."""
    _default_color = (.3, .3, .3, 1.)

    def __init__(self, color=None):
        super(LineVisual, self).__init__()
        self.set_shader('line')
        self.set_primitive_type('lines')
        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def validate(pos=None, color=None,
                 data_bounds=None,
                 ):
        assert pos is not None
        pos = np.atleast_2d(pos)
        assert pos.ndim == 2
        n_lines = pos.shape[0]
        assert pos.shape[1] == 4

        # Color.
        color = _get_array(color, (n_lines, 4), LineVisual._default_color)

        # By default, we assume that the coordinates are in NDC.
        if data_bounds is None:
            data_bounds = NDC
        data_bounds = _get_data_bounds(data_bounds, length=n_lines)
        data_bounds = data_bounds.astype(np.float64)
        assert data_bounds.shape == (n_lines, 4)

        return Bunch(pos=pos, color=color, data_bounds=data_bounds)

    @staticmethod
    def vertex_count(pos=None, **kwargs):
        """Take the output of validate() as input."""
        return pos.shape[0] * 2

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        pos = data.pos
        assert pos.ndim == 2
        assert pos.shape[1] == 4
        assert pos.dtype == np.float64
        n_lines = pos.shape[0]
        n_vertices = 2 * n_lines
        pos = pos.reshape((-1, 2))

        # Transform the positions.
        data_bounds = np.repeat(data.data_bounds, 2, axis=0)
        self.data_range.from_bounds = data_bounds
        pos_tr = self.transforms.apply(pos)

        # Position.
        assert pos_tr.shape == (n_vertices, 2)
        self.program['a_position'] = pos_tr.astype(np.float32)

        # Color.
        color = np.repeat(data.color, 2, axis=0)
        self.program['a_color'] = color.astype(np.float32)


class PolygonVisual(BaseVisual):
    """Polygon."""
    _default_color = (.5, .5, .5, 1.)

    def __init__(self):
        super(PolygonVisual, self).__init__()
        self.set_shader('polygon')
        self.set_primitive_type('line_loop')
        self.data_range = Range(NDC)
        self.transforms.add_on_cpu(self.data_range)

    @staticmethod
    def validate(pos=None,
                 data_bounds=None,
                 ):
        assert pos is not None
        pos = np.atleast_2d(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2

        # By default, we assume that the coordinates are in NDC.
        if data_bounds is None:
            data_bounds = NDC
        data_bounds = _get_data_bounds(data_bounds)
        data_bounds = data_bounds.astype(np.float64)
        assert data_bounds.shape == (1, 4)

        return Bunch(pos=pos, data_bounds=data_bounds)

    @staticmethod
    def vertex_count(pos=None, **kwargs):
        """Take the output of validate() as input."""
        return pos.shape[0]

    def set_data(self, *args, **kwargs):
        data = self.validate(*args, **kwargs)
        pos = data.pos
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        assert pos.dtype == np.float64
        n_vertices = pos.shape[0]

        # Transform the positions.
        self.data_range.from_bounds = data.data_bounds
        pos_tr = self.transforms.apply(pos)

        # Position.
        assert pos_tr.shape == (n_vertices, 2)
        self.program['a_position'] = pos_tr.astype(np.float32)

        self.program['u_color'] = self._default_color
