# -*- coding: utf-8 -*-

"""Common visuals.

All visuals derive from the base class `BaseVisual()`. They all follow the same structure.
Constant parameters are passed to the constructor. Variable parameters are passed to `set_data()`
which is the main method: it updates the OpenGL objects to update the graphics.
The `validate()` method is used to fill default values and validate the requested data.

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import gzip
from pathlib import Path

import numpy as np

from .base import BaseVisual
from .gloo import gl
from .transform import NDC
from .utils import (
    _tesselate_histogram, _get_texture, _get_array, _get_pos, _get_index)
from phy.gui.qt import is_high_dpi
from phylib.io.array import _as_array
from phylib.utils import Bunch
from phylib.utils.geometry import _get_data_bounds


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

DEFAULT_COLOR = (0.03, 0.57, 0.98, .75)


#------------------------------------------------------------------------------
# Patch visual
#------------------------------------------------------------------------------

class PatchVisual(BaseVisual):
    """Patch visual, displaying an arbitrary filled shape.

    Constructor
    -----------

    marker : string (used for all points in the scatter visual)
        Default: disc. Can be one of: arrow, asterisk, chevron, clover, club, cross, diamond,
        disc, ellipse, hbar, heart, infinity, pin, ring, spade, square, tag, triangle, vbar

    Parameters
    ----------

    x : array-like (1D)
    y : array-like (1D)
    pos : array-like (2D)
    color : array-like (2D, shape[1] == 4)
    primitive_type : str
        triangles, triangle_fan, triangle_strip
    depth : array-like (1D)
    data_bounds : array-like (2D, shape[1] == 4)

    """
    default_color = DEFAULT_COLOR

    def __init__(self, primitive_type='triangle_fan'):
        super(PatchVisual, self).__init__()
        self.set_shader('patch')
        self.set_primitive_type(primitive_type)
        self.set_data_range(NDC)

    def vertex_count(self, x=None, y=None, pos=None, **kwargs):
        """Number of vertices for the requested data."""
        return y.size if y is not None else len(pos)

    def validate(
            self, x=None, y=None, pos=None, color=None, depth=None,
            data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        if pos is None:
            x, y = _get_pos(x, y)
            pos = np.c_[x, y]
        pos = np.asarray(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        # Validate the data.
        color = _get_array(color, (n, 4), ScatterVisual.default_color, dtype=np.float32)
        depth = _get_array(depth, (n, 1), 0)
        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, pos)
            assert data_bounds.shape[0] == n

        return Bunch(
            pos=pos, color=color, depth=depth, data_bounds=data_bounds,
            _n_items=n, _n_vertices=n)

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)
        if data.data_bounds is not None:
            self.data_range.from_bounds = data.data_bounds
            pos_tr = self.transforms.apply(data.pos)
        else:
            pos_tr = data.pos
        pos_tr = np.c_[pos_tr, data.depth]
        self.program['a_position'] = pos_tr.astype(np.float32)
        self.program['a_color'] = data.color.astype(np.float32)
        self.emit_visual_set_data()
        return data

    def set_color(self, color):
        """Change the color of the markers."""
        color = _get_array(color, (self.n_vertices, 4), PatchVisual.default_color)
        self.program['a_color'] = color.astype(np.float32)


#------------------------------------------------------------------------------
# Scatter visuals
#------------------------------------------------------------------------------

class ScatterVisual(BaseVisual):
    """Scatter visual, displaying a fixed marker at various positions, colors, and marker sizes.

    Constructor
    -----------

    marker : string (used for all points in the scatter visual)
        Default: disc. Can be one of: arrow, asterisk, chevron, clover, club, cross, diamond,
        disc, ellipse, hbar, heart, infinity, pin, ring, spade, square, tag, triangle, vbar

    Parameters
    ----------

    x : array-like (1D)
    y : array-like (1D)
    pos : array-like (2D)
    color : array-like (2D, shape[1] == 4)
    size : array-like (1D)
        Marker sizes, in pixels
    depth : array-like (1D)
    data_bounds : array-like (2D, shape[1] == 4)

    """
    _init_keywords = ('marker',)
    default_marker_size = 10.
    default_marker = 'disc'
    default_color = DEFAULT_COLOR
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

    def __init__(self, marker=None, marker_scaling=None):
        super(ScatterVisual, self).__init__()

        # Set the marker type.
        self.marker = marker or self.default_marker
        assert self.marker in self._supported_markers

        self.set_shader('scatter')
        marker_scaling = marker_scaling or 'float marker_size = v_size;'
        self.fragment_shader = self.fragment_shader.replace('%MARKER_SCALING', marker_scaling)
        self.fragment_shader = self.fragment_shader.replace('%MARKER', self.marker)
        self.set_primitive_type('points')
        self.set_data_range(NDC)

    def vertex_count(self, x=None, y=None, pos=None, **kwargs):
        """Number of vertices for the requested data."""
        return y.size if y is not None else len(pos)

    def validate(
            self, x=None, y=None, pos=None, color=None, size=None, depth=None,
            data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        if pos is None:
            x, y = _get_pos(x, y)
            pos = np.c_[x, y]
        pos = np.asarray(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        # Validate the data.
        color = _get_array(color, (n, 4), ScatterVisual.default_color, dtype=np.float32)
        size = _get_array(size, (n, 1), ScatterVisual.default_marker_size)
        depth = _get_array(depth, (n, 1), 0)
        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, pos)
            assert data_bounds.shape[0] == n

        return Bunch(
            pos=pos, color=color, size=size, depth=depth, data_bounds=data_bounds,
            _n_items=n, _n_vertices=n)

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)
        if data.data_bounds is not None:
            self.data_range.from_bounds = data.data_bounds
            pos_tr = self.transforms.apply(data.pos)
        else:
            pos_tr = data.pos
        pos_tr = np.c_[pos_tr, data.depth]
        self.program['a_position'] = pos_tr.astype(np.float32)
        self.program['a_size'] = data.size.astype(np.float32)
        self.program['a_color'] = data.color.astype(np.float32)
        self.emit_visual_set_data()
        return data

    def set_color(self, color):
        """Change the color of the markers."""
        color = _get_array(color, (self.n_vertices, 4), ScatterVisual.default_color)
        self.program['a_color'] = color.astype(np.float32)

    def set_marker_size(self, marker_size):
        """Change the size of the markers."""
        size = _get_array(marker_size, (self.n_vertices, 1))
        assert np.all(size > 0)
        self.program['a_size'] = size.astype(np.float32)


class UniformScatterVisual(BaseVisual):
    """Scatter visual with a fixed marker color and size.

    Constructor
    -----------

    marker : str
    color : 4-tuple
    size : scalar

    Parameters
    ----------

    x : array-like (1D)
    y : array-like (1D)
    pos : array-like (2D)
    masks : array-like (1D)
        Similar to an alpha channel, but for color saturation instead of transparency.
    data_bounds : array-like (2D, shape[1] == 4)

    """

    _init_keywords = ('marker', 'color', 'size')
    default_marker_size = 10.
    default_marker = 'disc'
    default_color = DEFAULT_COLOR
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
        self.marker = marker or self.default_marker
        assert self.marker in self._supported_markers

        self.set_shader('uni_scatter')
        self.fragment_shader = self.fragment_shader.replace('%MARKER', self.marker)

        self.color = color or self.default_color
        self.marker_size = size or self.default_marker_size

        self.set_primitive_type('points')
        self.set_data_range(NDC)

    def vertex_count(self, x=None, y=None, pos=None, **kwargs):
        """Number of vertices for the requested data."""
        return y.size if y is not None else len(pos)

    def validate(self, x=None, y=None, pos=None, masks=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        if pos is None:
            x, y = _get_pos(x, y)
            pos = np.c_[x, y]
        pos = np.asarray(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n = pos.shape[0]

        masks = _get_array(masks, (n, 1), 1., np.float32)
        assert masks.shape == (n, 1)

        # The mask is clu_idx + fractional mask
        masks *= .99999

        # Validate the data.
        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, pos)
            assert data_bounds.shape[0] == n

        return Bunch(pos=pos, masks=masks, data_bounds=data_bounds, _n_items=n, _n_vertices=n)

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)
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
        self.emit_visual_set_data()
        return data


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
    """Minimum of an array, return 0 on empty arrays."""
    return arr.min() if arr is not None and len(arr) > 0 else 0


def _max(arr):
    """Maximum of an array, return 1 on empty arrays."""
    return arr.max() if arr is not None and len(arr) > 0 else 1


class PlotVisual(BaseVisual):
    """Plot visual, with multiple line plots of various sizes and colors.

    Parameters
    ----------

    x : array-like (1D), or list of 1D arrays for different plots
    y : array-like (1D), or list of 1D arrays, for different plots
    color : array-like (2D, shape[-1] == 4)
    depth : array-like (1D)
    masks : array-like (1D)
        Similar to an alpha channel, but for color saturation instead of transparency.
    data_bounds : array-like (2D, shape[1] == 4)

    """

    default_color = DEFAULT_COLOR
    _noconcat = ('x', 'y')

    def __init__(self):
        super(PlotVisual, self).__init__()

        self.set_shader('plot')
        self.set_primitive_type('line_strip')
        self.set_data_range(NDC)

    def validate(
            self, x=None, y=None, color=None, depth=None, masks=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""

        assert y is not None
        y = _as_list(y)

        if x is None:
            x = [np.linspace(-1., 1., len(_)) for _ in y]
        x = _as_list(x)

        # Remove empty elements.
        assert len(x) == len(y)

        assert [len(_) for _ in x] == [len(_) for _ in y]

        n_signals = len(x)

        if isinstance(data_bounds, str) and data_bounds == 'auto':
            xmin = [_min(_) for _ in x]
            ymin = [_min(_) for _ in y]
            xmax = [_max(_) for _ in x]
            ymax = [_max(_) for _ in y]
            data_bounds = np.c_[xmin, ymin, xmax, ymax]

        color = _get_array(color, (n_signals, 4),
                           PlotVisual.default_color,
                           dtype=np.float32,
                           )
        assert color.shape == (n_signals, 4)

        masks = _get_array(masks, (n_signals, 1), 1., np.float32)
        # The mask is clu_idx + fractional mask
        masks *= .99999
        assert masks.shape == (n_signals, 1)

        depth = _get_array(depth, (n_signals, 1), 0)
        assert depth.shape == (n_signals, 1)

        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, length=n_signals)
            data_bounds = data_bounds.astype(np.float64)
            assert data_bounds.shape == (n_signals, 4)

        return Bunch(
            x=x, y=y, color=color, depth=depth, data_bounds=data_bounds, masks=masks,
            _n_items=n_signals, _n_vertices=self.vertex_count(y=y))

    def set_color(self, color):
        """Update the visual's color."""
        assert color.shape == (self.n_vertices, 4)
        self.program['a_color'] = color.astype(np.float32)

    def vertex_count(self, y=None, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        return y.size if isinstance(y, np.ndarray) else sum(len(_) for _ in y)

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)

        assert isinstance(data.y, list)
        n_signals = len(data.y)
        n_samples = [len(_) for _ in data.y]

        self.n_signals = n_signals
        self.n_samples = n_samples

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

        # Masks.
        masks = np.repeat(data.masks, n_samples, axis=0)
        assert masks.shape == (n, 1)

        # Position and depth.
        depth = np.repeat(data.depth, n_samples, axis=0)
        pos_depth = np.c_[pos, depth]

        self.program['a_position'] = pos_depth.astype(np.float32)
        self.program['a_color'] = color.astype(np.float32)
        self.program['a_signal_index'] = signal_index.astype(np.float32)
        self.program['a_mask'] = masks.astype(np.float32)
        self.program['u_mask_max'] = _max(masks)

        self.emit_visual_set_data()
        return data


class UniformPlotVisual(BaseVisual):
    """A plot visual with a uniform color.

    Constructor
    -----------

    color : 4-tuple
    depth : scalar

    Parameters
    ----------

    x : array-like (1D), or list of 1D arrays for different plots
    y : array-like (1D), or list of 1D arrays, for different plots
    masks : array-like (1D)
        Similar to an alpha channel, but for color saturation instead of transparency.
    data_bounds : array-like (2D, shape[1] == 4)

    """

    default_color = DEFAULT_COLOR
    _noconcat = ('x', 'y')

    def __init__(self, color=None, depth=None):
        super(UniformPlotVisual, self).__init__()

        self.set_shader('uni_plot')
        self.set_primitive_type('line_strip')
        self.color = color or self.default_color
        self.set_data_range(NDC)

    def validate(self, x=None, y=None, masks=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""

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
        # The mask is clu_idx + fractional mask
        masks *= .99999
        assert masks.shape == (n_signals, 1)

        if isinstance(data_bounds, str) and data_bounds == 'auto':
            xmin = [_min(_) for _ in x]
            ymin = [_min(_) for _ in y]
            xmax = [_max(_) for _ in x]
            ymax = [_max(_) for _ in y]
            data_bounds = np.c_[xmin, ymin, xmax, ymax]

        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, length=n_signals)
            data_bounds = data_bounds.astype(np.float64)
            assert data_bounds.shape == (n_signals, 4)

        return Bunch(
            x=x, y=y, masks=masks, data_bounds=data_bounds,
            _n_items=n_signals, _n_vertices=self.vertex_count(y=y))

    def vertex_count(self, y=None, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        return y.size if isinstance(y, np.ndarray) else sum(len(_) for _ in y)

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)

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

        self.emit_visual_set_data()
        return data


#------------------------------------------------------------------------------
# Histogram visual
#------------------------------------------------------------------------------

class HistogramVisual(BaseVisual):
    """A histogram visual.

    Parameters
    ----------

    hist : array-like (1D), or list of 1D arrays, or 2D array
    color : array-like (2D, shape[1] == 4)
    ylim : array-like (1D)
        The maximum hist value in the viewport.

    """

    default_color = DEFAULT_COLOR

    def __init__(self):
        super(HistogramVisual, self).__init__()

        self.set_shader('histogram')
        self.set_primitive_type('triangles')
        self.set_data_range([0, 0, 1, 1])

    def validate(self, hist=None, color=None, ylim=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        assert hist is not None
        hist = np.asarray(hist, np.float64)
        if hist.ndim == 1:
            hist = hist[None, :]
        assert hist.ndim == 2
        n_hists, n_bins = hist.shape

        # Validate the data.
        color = _get_array(color, (n_hists, 4), HistogramVisual.default_color, dtype=np.float32)

        # Validate ylim.
        if ylim is None:
            ylim = hist.max() if hist.size > 0 else 1.
        ylim = np.atleast_1d(ylim)
        if len(ylim) == 1:
            ylim = np.tile(ylim, n_hists)
        if ylim.ndim == 1:
            ylim = ylim[:, np.newaxis]
        assert ylim.shape == (n_hists, 1)

        return Bunch(
            hist=hist, ylim=ylim, color=color,
            _n_items=n_hists, _n_vertices=self.vertex_count(hist))

    def vertex_count(self, hist, **kwargs):
        """Number of vertices for the requested data."""
        hist = np.atleast_2d(hist)
        n_hists, n_bins = hist.shape
        return 6 * n_hists * n_bins

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)
        hist = data.hist

        n_hists, n_bins = hist.shape
        n = self.vertex_count(hist)

        # NOTE: this must be set *before* `apply_cpu_transforms` such
        # that the histogram is correctly normalized.
        data_bounds = np.c_[np.zeros((n_hists, 2)), n_bins * np.ones((n_hists, 1)), data.ylim]
        data_bounds = np.repeat(data_bounds, 6 * n_bins, axis=0)
        self.data_range.from_bounds = data_bounds

        # Set the transformed position.
        pos = np.vstack([_tesselate_histogram(row) for row in hist])
        pos_tr = self.transforms.apply(pos)
        assert pos_tr.shape == (n, 2)
        self.program['a_position'] = pos_tr.astype(np.float32)

        # Generate the hist index.
        hist_index = _get_index(n_hists, n_bins * 6, n)
        self.program['a_hist_index'] = hist_index.astype(np.float32)

        # Hist colors.
        tex = _get_texture(data.color, self.default_color, n_hists, [0, 1])
        self.program['u_color'] = tex.astype(np.float32)
        self.program['n_hists'] = n_hists

        self.emit_visual_set_data()
        return data


#------------------------------------------------------------------------------
# Test visual
#------------------------------------------------------------------------------

FONT_MAP_PATH = Path(__file__).parent / 'static/SourceCodePro-Regular.npy.gz'
FONT_MAP_SIZE = (6, 16)
SDF_SIZE = 64
GLYPH_SIZE = (40, 64)
FONT_MAP_CHARS = ''.join(chr(i) for i in range(32, 32 + FONT_MAP_SIZE[0] * FONT_MAP_SIZE[1]))


class TextVisual(BaseVisual):
    """Display strings at multiple locations.

    Constructor
    -----------

    color : 4-tuple
    font_size : float
        The font size, in points (8 by default).

    Parameters
    ----------

    pos : array-like (2D)
        Position of each string (of variable length).
    text : list of strings (variable lengths)
    anchor : array-like (2D)
        For each string, specifies the anchor of the string with respect to the string's position.

        Examples:

        * (0, 0): text centered at the position
        * (1, 1): the position is at the lower left of the string
        * (1, -1): the position is at the upper left of the string
        * (-1, 1): the position is at the lower right of the string
        * (-1, -1): the position is at the upper right of the string

        Values higher than 1 or lower than -1 can be used as margins, knowing that the unit
        of the anchor is (string width, string height).

    data_bounds : array-like (2D, shape[1] == 4)

    """
    default_color = (1., 1., 1., 1.)
    default_font_size = 6.
    _init_keywords = ('color',)
    _noconcat = ('text',)

    def __init__(self, color=None, font_size=None):
        super(TextVisual, self).__init__()
        self.set_shader('msdf')
        self.set_primitive_type('triangles')
        self.set_data_range(NDC)

        # Color.
        color = color if color is not None else TextVisual.default_color
        assert not isinstance(color, np.ndarray)  # uniform color for now
        assert len(color) == 4
        self.color = color

        # Font size.
        self.font_size = font_size or self.default_font_size  # in points
        if is_high_dpi():  # pragma: no cover
            self.font_size *= 2
        assert self.font_size > 0

        # Load the multi signed distance field font map.
        with gzip.open(str(FONT_MAP_PATH), 'rb') as f:
            self._tex = np.load(f)

    def _get_glyph_indices(self, s):
        return [FONT_MAP_CHARS.index(char) for char in s]

    def validate(
            self, pos=None, text=None, color=None, anchor=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""

        if text is None:
            text = []
        if isinstance(text, str):
            text = [text]
        if pos is None:
            pos = np.zeros((len(text), 2))

        assert pos is not None
        pos = np.atleast_2d(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        n_text = pos.shape[0]
        assert len(text) == n_text

        color = color if color is not None else self.default_color
        color = np.atleast_2d(color)
        if color.shape[0] == 1:
            color = np.repeat(color, n_text, axis=0)
        assert color.ndim == 2
        assert color.shape[1] == 4
        assert len(color) == n_text

        anchor = anchor if anchor is not None else (0., 0.)
        anchor = np.atleast_2d(anchor)
        if anchor.shape[0] == 1:
            anchor = np.repeat(anchor, n_text, axis=0)
        assert anchor.ndim == 2
        assert anchor.shape == (n_text, 2)

        data_bounds = data_bounds if data_bounds is not None else NDC
        data_bounds = _get_data_bounds(data_bounds, pos)
        assert data_bounds.shape[0] == n_text
        data_bounds = data_bounds.astype(np.float64)
        assert data_bounds.shape == (n_text, 4)

        return Bunch(
            pos=pos, text=text, anchor=anchor, data_bounds=data_bounds, color=color,
            _n_items=n_text, _n_vertices=self.vertex_count(text=text))

    def vertex_count(self, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        # Total number of glyphs * 6 (6 vertices per glyph).
        return sum(map(len, kwargs.get('text', ''))) * 6

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)

        pos = data.pos.astype(np.float64)
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        assert pos.dtype == np.float64

        # Concatenate all strings.
        text = data.text
        n_text = len(text)
        lengths = list(map(len, text))
        assert isinstance(text, list)
        text = ''.join(text)
        a_char_index = self._get_glyph_indices(text)
        n_glyphs = len(a_char_index)

        tex = self._tex
        glyph_height = tex.shape[0] // 6
        glyph_width = tex.shape[1] // 16
        glyph_size = (glyph_width * self.font_size / 12, glyph_height * self.font_size / 12)

        # Position of all glyphs.
        a_position = np.repeat(pos, lengths, axis=0)
        if not len(lengths):
            a_glyph_index = np.zeros((0,))
        else:
            a_glyph_index = np.concatenate([np.arange(n) for n in lengths])
        a_quad_index = np.arange(6)

        a_position = np.repeat(a_position, 6, axis=0)
        a_glyph_index = np.repeat(a_glyph_index, 6)
        a_quad_index = np.tile(a_quad_index, n_glyphs)
        a_char_index = np.repeat(a_char_index, 6)

        a_anchor = np.repeat(data.anchor, lengths, axis=0)
        a_anchor = np.repeat(a_anchor, 6, axis=0)

        a_color = np.repeat(data.color, lengths, axis=0)
        a_color = np.repeat(a_color, 6, axis=0)

        a_lengths = np.repeat(lengths, lengths)
        a_lengths = np.repeat(a_lengths, 6)

        a_string_index = np.repeat(np.arange(n_text), lengths)
        a_string_index = np.repeat(a_string_index, 6)

        n_vertices = n_glyphs * 6

        # Transform the positions.
        assert data.data_bounds is not None
        data_bounds = data.data_bounds
        data_bounds = np.repeat(data_bounds, lengths, axis=0)
        data_bounds = np.repeat(data_bounds, 6, axis=0)
        assert data_bounds.shape == (n_vertices, 4)
        self.data_range.from_bounds = data_bounds
        pos_tr = self.transforms.apply(a_position)
        assert pos_tr.shape == (n_vertices, 2)

        assert a_glyph_index.shape == (n_vertices,)  # 000000111111...
        assert a_quad_index.shape == (n_vertices,)  # 012345012345....
        assert a_char_index.shape == (n_vertices,)  # 67.67.67.67.67.67.97.97.97.97.97...
        assert a_anchor.shape == (n_vertices, 2)  # (1, 1), (1, 1), ...
        assert a_color.shape == (n_vertices, 4)
        assert a_lengths.shape == (n_vertices,)  # 7777777777777777777...
        assert a_string_index.shape == (n_vertices,)  # 000000000000000111111111111...

        self.program['a_position'] = pos_tr.astype(np.float32)
        self.program['a_color'] = a_color.astype(np.float32)
        self.program['a_glyph_index'] = a_glyph_index.astype(np.float32)
        self.program['a_quad_index'] = a_quad_index.astype(np.float32)
        self.program['a_char_index'] = a_char_index.astype(np.float32)
        self.program['a_anchor'] = a_anchor.astype(np.float32)
        self.program['a_lengths'] = a_lengths.astype(np.float32)
        self.program['a_string_index'] = a_string_index.astype(np.float32)

        self.program['u_glyph_size'] = glyph_size
        self.program['u_color'] = self.color

        self.program['u_tex'] = tex[::-1, :]
        self.program['u_tex_size'] = tex.shape[:2]

        self.emit_visual_set_data()
        return data

    def on_draw(self):
        # NOTE: use linear interpolation for the SDF texture.
        self.program._uniforms['u_tex']._data.set_interpolation('linear')
        super(TextVisual, self).on_draw()


#------------------------------------------------------------------------------
# Line visual
#------------------------------------------------------------------------------

class LineVisual(BaseVisual):
    """Line segments.

    Parameters
    ----------
    pos : array-like (2D)
    color : array-like (2D, shape[1] == 4)
    data_bounds : array-like (2D, shape[1] == 4)

    """

    default_color = (.3, .3, .3, 1.)
    _init_keywords = ('color',)

    def __init__(self):
        super(LineVisual, self).__init__()
        self.set_shader('line')
        self.set_primitive_type('lines')
        self.set_data_range(NDC)

    def validate(self, pos=None, color=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        assert pos is not None
        pos = _as_array(pos)
        pos = np.atleast_2d(pos)
        assert pos.ndim == 2
        n_lines = pos.shape[0]
        assert pos.shape[1] == 4

        # Color.
        color = _get_array(color, (n_lines, 4), LineVisual.default_color)

        # By default, we assume that the coordinates are in NDC.
        if data_bounds is None:
            data_bounds = NDC
        data_bounds = _get_data_bounds(data_bounds, length=n_lines)
        data_bounds = data_bounds.astype(np.float64)
        assert data_bounds.shape == (n_lines, 4)

        return Bunch(
            pos=pos, color=color, data_bounds=data_bounds,
            _n_items=n_lines, _n_vertices=self.vertex_count(pos=pos))

    def vertex_count(self, pos=None, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        pos = np.atleast_2d(pos)
        assert pos.shape[1] == 4
        return pos.shape[0] * 2

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)

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

        self.emit_visual_set_data()
        return data


#------------------------------------------------------------------------------
# Agg line visual
#------------------------------------------------------------------------------

def ortho(left, right, bottom, top, znear, zfar):  # pragma: no cover
    """Create orthographic projection matrix

    Parameters
    ----------
    left : float
        Left coordinate of the field of view.
    right : float
        Right coordinate of the field of view.
    bottom : float
        Bottom coordinate of the field of view.
    top : float
        Top coordinate of the field of view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.

    Returns
    -------
    M : array
        Orthographic projection matrix (4x4).
    """
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / (right - left)
    M[3, 0] = -(right + left) / float(right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[3, 1] = -(top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M


class LineAggGeomVisual(BaseVisual):  # pragma: no cover
    """[experimental] Line agg using geometry shader. Not currently used. Uses non-standard
    geometry shader extension, may only work on NVIDIA.

    TODO: fix pan zoom which is currently broken because of viewport coordinate transform.

    Parameters
    ----------
    pos : array-like (2D)
    color : array-like (2D, shape[1] == 4)
    data_bounds : array-like (2D, shape[1] == 4)

    """

    default_color = (.75, .75, .75, 1.)
    _init_keywords = ('color',)

    def __init__(self):
        super(LineAggGeomVisual, self).__init__()
        self.set_shader('line_agg_geom')
        self.set_primitive_type('line_strip_adjacency_ext')
        # Geometry shader params.
        self.geometry_count = 4
        self.geometry_in = gl.GL_LINES_ADJACENCY_EXT
        self.geometry_out = gl.GL_TRIANGLE_STRIP
        self.set_data_range(NDC)

    def _get_index_buffer(self, P, closed=True):
        if closed:
            if np.allclose(P[0], P[1]):
                I = (np.arange(len(P) + 2) - 1)
                I[0], I[-1] = 0, len(P) - 1
            else:
                I = (np.arange(len(P) + 3) - 1)
                I[0], I[-2], I[-1] = len(P) - 1, 0, 1
        else:
            I = (np.arange(len(P) + 2) - 1)
            I[0], I[-1] = 0, len(P) - 1
        return I

    def validate(self, pos=None, color=None, line_width=10., data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        assert pos is not None
        pos = _as_array(pos)
        pos = np.atleast_2d(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 2

        line_width = np.clip(line_width, 5, 100)

        # Color.
        color = _get_array(color, (1, 4), self.default_color)

        # By default, we assume that the coordinates are in NDC.
        if data_bounds is None:
            data_bounds = NDC
        data_bounds = _get_data_bounds(data_bounds)
        data_bounds = data_bounds.astype(np.float64)
        # assert data_bounds.shape == (n_lines, 2)

        return Bunch(
            pos=pos, color=color, line_width=line_width, data_bounds=data_bounds,
            _n_items=1, _n_vertices=self.vertex_count(pos=pos))

    def vertex_count(self, pos=None, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        pos = np.atleast_2d(pos)
        assert pos.shape[1] == 2
        return pos.shape[0]

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)

        pos = data.pos
        assert pos.ndim == 2
        assert pos.shape[1] == 2
        assert pos.dtype == np.float64

        # Transform the positions.
        self.data_range.from_bounds = data.data_bounds
        pos_tr = self.transforms.apply(pos).astype(np.float32)

        self.program['position'] = pos_tr
        self.program['linewidth'] = data.line_width
        self.program['antialias'] = 1.0
        self.program['miter_limit'] = 4.0
        self.program['color'] = data.color

        self.index_buffer = self._get_index_buffer(pos_tr, closed=False)

        self.emit_visual_set_data()
        return data

    def on_resize(self, width, height):
        projection = ortho(0, width, 0, height, -1, +1)
        self.program['projection'] = projection


class PlotAggVisual(BaseVisual):
    """Plot agg visual, with multiple line plots of various sizes and colors.

    Parameters
    ----------

    x : array-like (1D), or list of 1D arrays for different plots
    y : array-like (1D), or list of 1D arrays, for different plots
    color : array-like (2D, shape[-1] == 4)
    depth : array-like (1D)
    masks : array-like (1D)
        Similar to an alpha channel, but for color saturation instead of transparency.
    data_bounds : array-like (2D, shape[1] == 4)

    """

    default_color = DEFAULT_COLOR
    default_line_width = 3.0
    _noconcat = ('x', 'y')

    def __init__(self, line_width=None, closed=False):
        super(PlotAggVisual, self).__init__()

        self.set_shader('plot_agg')
        self.set_primitive_type('triangle_strip')
        self.set_data_range(NDC)
        self.closed = closed
        self.line_width = line_width or self.default_line_width

    def validate(
            self, x=None, y=None, color=None, depth=None, masks=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""

        assert y is not None
        y = np.atleast_2d(_as_array(y))
        n_signals, n_samples = y.shape
        if x is None:
            x = np.tile(np.linspace(-1., 1., n_samples), (n_signals, 1))
        x = np.atleast_2d(_as_array(x))

        if isinstance(data_bounds, str) and data_bounds == 'auto':
            xmin, xmax = x.min(axis=1), x.max(axis=1)
            ymin, ymax = y.min(axis=1), y.max(axis=1)
            data_bounds = np.c_[xmin, ymin, xmax, ymax]

        color = _get_array(color, (n_signals, 4),
                           PlotVisual.default_color,
                           dtype=np.float32,
                           )
        assert color.shape == (n_signals, 4)

        masks = _get_array(masks, (n_signals, 1), 1., np.float32)
        # The mask is clu_idx + fractional mask
        masks *= .99999
        assert masks.shape == (n_signals, 1)

        depth = _get_array(depth, (n_signals, 1), 0)
        assert depth.shape == (n_signals, 1)

        if data_bounds is not None:
            data_bounds = _get_data_bounds(data_bounds, length=n_signals)
            data_bounds = data_bounds.astype(np.float64)
            assert data_bounds.shape == (n_signals, 4)

        return Bunch(
            x=x, y=y, color=color, depth=depth, masks=masks, data_bounds=data_bounds,
            _n_items=n_signals, _n_vertices=self.vertex_count(y=y))

    def vertex_count(self, y=None, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        itemcount, itemsize = y.shape
        if self.closed:
            return itemcount * 2 * (itemsize + 3)
        else:
            return itemcount * 2 * (itemsize + 2)

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)
        closed = self.closed

        n_signals, n_samples = data.y.shape
        n = data.y.size
        self.n_signals = n_signals
        self.n_samples = n_samples

        if n_signals * n_samples == 0:
            return data

        # Generate the position array.
        pos = np.c_[data.x.ravel(), data.y.ravel()]
        assert pos.shape == (n, 2)

        # Generate the color attribute.
        color = data.color
        assert color.shape == (n_signals, 4)
        color = np.tile(color.reshape((n_signals, 1, 4)), (1, n_samples, 1))
        assert color.shape == (n_signals, n_samples, 4)

        # Transform the positions.
        if data.data_bounds is not None:
            data_bounds = np.repeat(data.data_bounds, n_samples, axis=0)
            self.data_range.from_bounds = data_bounds
            pos = self.transforms.apply(pos)

        itemsize = self.n_samples
        itemcount = self.n_signals

        pos = np.c_[pos, np.zeros((len(pos), 1))].astype(np.float32)
        P = pos.reshape(itemcount, itemsize, 3)
        if self.closed:
            a_prev = np.empty((itemcount, itemsize + 3, 3), dtype=np.float32)
            a_curr = np.empty((itemcount, itemsize + 3, 3), dtype=np.float32)
            a_next = np.empty((itemcount, itemsize + 3, 3), dtype=np.float32)
            a_color = np.empty((itemcount, itemsize + 3, 4), dtype=np.float32)

            a_prev[:, 2:-1, :] = P
            a_prev[:, 1, :] = a_prev[:, -2, :]
            a_curr[:, 1:-2, :] = P
            a_curr[:, -2, :] = a_curr[:, 1, :]
            a_color[:, 1:-2, :] = color
            a_color[:, -2, :] = a_color[:, 1, :]
            a_next[:, 0:-3, :] = P
            a_next[:, -3, :] = a_next[:, 0, :]
            a_next[:, -2, :] = a_next[:, 1, :]
        else:
            a_prev = np.empty((itemcount, itemsize + 2, 3), dtype=np.float32)
            a_curr = np.empty((itemcount, itemsize + 2, 3), dtype=np.float32)
            a_next = np.empty((itemcount, itemsize + 2, 3), dtype=np.float32)
            a_color = np.empty((itemcount, itemsize + 2, 4), dtype=np.float32)

            a_prev[:, 2:, :] = P
            a_prev[:, 1, :] = a_prev[:, 2, :]
            a_curr[:, 1:-1, :] = P
            a_color[:, 1:-1, :] = color
            a_next[:, :-2, :] = P
            a_next[:, -2, :] = a_next[:, -3, :]

        tmp = {}
        for name in ('a_prev', 'a_curr', 'a_next', 'a_color'):
            V = locals()[name]
            assert V.ndim == 3
            last_dim = V.shape[-1]
            V[:, 0, ...] = V[:, 1, ...]
            V[:, -1, ...] = V[:, -2, ...]
            V = V.reshape((-1, last_dim))
            V = np.repeat(V, 2, axis=0)
            if closed:
                V = V.reshape((itemcount, 2 * (itemsize + 3), V.shape[-1]))
            else:
                V = V.reshape((itemcount, 2 * (itemsize + 2), last_dim))
            tmp[name] = V

        a_prev = tmp['a_prev']
        a_curr = tmp['a_curr']
        a_next = tmp['a_next']
        a_color = tmp['a_color']

        n_vertices_per_item = 2 * (itemsize + 2 + int(closed))
        N = itemcount * n_vertices_per_item

        a_id = np.tile([1, -1], N // 2)
        a_id = a_id.reshape(itemcount, n_vertices_per_item)
        a_id[:, :2] = 2, -2
        a_id[:, -2:] = 2, -2

        assert a_prev.size == N * 3
        assert a_curr.size == N * 3
        assert a_next.size == N * 3
        assert a_color.size == N * 4
        assert a_id.size == N

        # Masks.
        masks = np.repeat(data.masks, n_vertices_per_item, axis=0)
        assert masks.shape == (N, 1)

        # Position and depth.
        depth = np.repeat(data.depth, n_vertices_per_item, axis=0)
        assert depth.shape == (N, 1)

        self.program['a_prev'] = a_prev.astype(np.float32).ravel()
        self.program['a_curr'] = a_curr.astype(np.float32).ravel()
        self.program['a_next'] = a_next.astype(np.float32).ravel()
        self.program['a_id'] = a_id.astype(np.float32).ravel()
        self.program['a_color'] = a_color.astype(np.float32).ravel()
        self.program['a_mask'] = masks.astype(np.float32)
        self.program['a_depth'] = depth.astype(np.float32).ravel()
        self.program['u_mask_max'] = _max(masks)

        self.program['u_linewidth'] = self.line_width
        self.program['u_antialias'] = 1.0

        self.emit_visual_set_data()
        return data


#------------------------------------------------------------------------------
# Image visual
#------------------------------------------------------------------------------

class ImageVisual(BaseVisual):
    """Display a 2D image.

    Parameters
    ----------
    image : array-like (3D)

    """

    def __init__(self):
        super(ImageVisual, self).__init__()

        self.set_shader('image')
        self.set_primitive_type('triangles')

    def validate(self, image=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
        assert image is not None
        image = np.asarray(image, np.float32)
        assert image.ndim == 3
        assert image.shape[2] == 4
        return Bunch(image=image, _n_items=1, _n_vertices=self.vertex_count())

    def vertex_count(self, image=None, **kwargs):
        """Number of vertices for the requested data."""
        return 6

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)
        image = data.image

        pos = np.array([
            [-1, -1],
            [-1, +1],
            [+1, -1],
            [-1, +1],
            [+1, +1],
            [+1, -1],
        ])
        tex_coords = np.array([
            [0, 1],
            [0, 0],
            [+1, 1],
            [0, 0],
            [+1, 0],
            [+1, 1],
        ])
        self.program['a_position'] = pos.astype(np.float32)
        self.program['a_tex_coords'] = tex_coords.astype(np.float32)
        self.program['u_tex'] = image.astype(np.float32)

        self.emit_visual_set_data()
        return data


#------------------------------------------------------------------------------
# Polygon visual
#------------------------------------------------------------------------------

class PolygonVisual(BaseVisual):
    """Polygon.

    Parameters
    ----------
    pos : array-like (2D)
    data_bounds : array-like (2D, shape[1] == 4)

    """
    default_color = (1, 1, 1, 1)

    def __init__(self):
        super(PolygonVisual, self).__init__()
        self.set_shader('polygon')
        self.set_primitive_type('line_loop')
        self.set_data_range(NDC)

    def validate(self, pos=None, data_bounds=None, **kwargs):
        """Validate the requested data before passing it to set_data()."""
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

        return Bunch(
            pos=pos, data_bounds=data_bounds,
            _n_items=pos.shape[0], _n_vertices=self.vertex_count(pos=pos))

    def vertex_count(self, pos=None, **kwargs):
        """Number of vertices for the requested data."""
        """Take the output of validate() as input."""
        return pos.shape[0]

    def set_data(self, *args, **kwargs):
        """Update the visual data."""
        data = self.validate(*args, **kwargs)
        self.n_vertices = self.vertex_count(**data)

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

        self.program['u_color'] = self.default_color

        self.emit_visual_set_data()
        return data
