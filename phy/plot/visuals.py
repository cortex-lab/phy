# -*- coding: utf-8 -*-

"""Common visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy.gloo import Texture2D

from .base import BaseVisual
from .transform import Range, NDC
from .utils import (_enable_depth_mask,
                    _tesselate_histogram,
                    _get_texture,
                    _get_array,
                    _get_data_bounds,
                    _get_pos_depth,
                    _check_pos_2D,
                    _get_index,
                    _get_linear_x,
                    _get_hist_max,
                    _get_color,
                    )


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
        # Default bounds.
        self.data_bounds = NDC
        self.n_points = None

        # Set the marker type.
        self.marker = marker or self._default_marker
        assert self.marker in self._supported_markers

        # Enable transparency.
        _enable_depth_mask()

        self.set_shader('scatter')
        self.fragment_shader = self.fragment_shader.replace('%MARKER',
                                                            self.marker)
        self.set_primitive_type('points')
        self.transforms.add_on_cpu(Range(from_bounds=self.data_bounds))

    def set_data(self,
                 pos=None,
                 depth=None,
                 color=None,
                 marker=None,
                 size=None,
                 data_bounds=None,
                 ):
        pos = _check_pos_2D(pos)
        n = pos.shape[0]
        assert pos.shape == (n, 2)

        # Set the data bounds from the data.
        self.data_bounds = _get_data_bounds(data_bounds, pos)

        pos_tr = self.transforms.apply(pos)
        self.program['a_position'] = _get_pos_depth(pos_tr, depth)
        self.program['a_size'] = _get_array(size, (n, 1),
                                            self._default_marker_size)
        self.program['a_color'] = _get_array(color, (n, 4),
                                             self._default_color)


class PlotVisual(BaseVisual):
    _default_color = DEFAULT_COLOR

    def __init__(self, n_samples=None):
        super(PlotVisual, self).__init__()
        self.data_bounds = NDC
        self.n_samples = n_samples
        _enable_depth_mask()

        self.set_shader('plot')
        self.set_primitive_type('line_strip')
        self.transforms.add_on_cpu(Range(from_bounds=self.data_bounds))

    def set_data(self,
                 x=None,
                 y=None,
                 depth=None,
                 data_bounds=None,
                 plot_colors=None,
                 ):

        # Default x coordinates.
        if x is None:
            assert y is not None
            x = _get_linear_x(*y.shape)

        assert x is not None
        assert y is not None
        assert x.ndim == 2
        assert x.shape == y.shape
        n_signals, n_samples = x.shape
        if self.n_samples:
            assert n_samples == self.n_samples
        n = n_signals * n_samples

        # Generate the (n, 2) pos array.
        pos = np.empty((n, 2), dtype=np.float32)
        pos[:, 0] = x.ravel()
        pos[:, 1] = y.ravel()
        pos = _check_pos_2D(pos)

        self.data_bounds = _get_data_bounds(data_bounds, pos)

        # Set the transformed position.
        pos_tr = self.transforms.apply(pos)

        # Depth.
        depth = _get_array(depth, (n_signals,), 0)
        depth = np.repeat(depth, n_samples)
        self.program['a_position'] = _get_pos_depth(pos_tr, depth)

        # Generate the signal index.
        self.program['a_signal_index'] = _get_index(n_signals, n_samples, n)

        # Signal colors.
        plot_colors = _get_texture(plot_colors, self._default_color,
                                   n_signals, [0, 1])
        self.program['u_plot_colors'] = Texture2D(plot_colors)

        # Number of signals.
        self.program['n_signals'] = n_signals


class HistogramVisual(BaseVisual):
    _default_color = DEFAULT_COLOR

    def __init__(self):
        super(HistogramVisual, self).__init__()
        self.n_bins = 0
        self.hist_max = 1

        self.set_shader('histogram')
        self.set_primitive_type('triangles')
        self.transforms.add_on_cpu(Range(from_bounds=[0, 0, self.n_bins,
                                                      self.hist_max],
                                         to_bounds=[0, 0, 1, 1]))
        # (0, 0, 1, v)
        self.transforms.add_on_gpu(Range(from_bounds='hist_bounds',
                                         to_bounds=NDC))

    def set_data(self,
                 hist=None,
                 ylim=None,
                 color=None,
                 ):
        hist = _check_pos_2D(hist)
        n_hists, n_bins = hist.shape
        n = 6 * n_hists * n_bins
        # Store n_bins for get_transforms().
        self.n_bins = n_bins

        # NOTE: this must be set *before* `apply_cpu_transforms` such
        # that the histogram is correctly normalized.
        self.hist_max = _get_hist_max(hist)

        # Set the transformed position.
        pos = np.vstack(_tesselate_histogram(row) for row in hist)
        pos_tr = self.transforms.apply(pos)
        pos_tr = np.asarray(pos_tr, dtype=np.float32)
        assert pos_tr.shape == (n, 2)
        self.program['a_position'] = pos_tr

        # Generate the hist index.
        self.program['a_hist_index'] = _get_index(n_hists, n_bins * 6, n)

        # Hist colors.
        self.program['u_color'] = _get_texture(color,
                                               self._default_color,
                                               n_hists, [0, 1])

        # Hist bounds.
        assert ylim is None or len(ylim) == n_hists
        hist_bounds = np.c_[np.zeros((n_hists, 2)),
                            np.ones((n_hists, 1)),
                            ylim / self.hist_max] if ylim is not None else None
        hist_bounds = _get_texture(hist_bounds, [0, 0, 1, 1],
                                   n_hists, [0, 10])
        self.program['u_hist_bounds'] = Texture2D(hist_bounds)
        self.program['n_hists'] = n_hists


class TextVisual(BaseVisual):
    def __init__(self):
        super(TextVisual, self).__init__()
        self.set_shader('text')
        self.set_primitive_type('points')

    def set_data(self):
        pass


class BoxVisual(BaseVisual):
    _default_color = (.35, .35, .35, 1.)

    def __init__(self):
        super(BoxVisual, self).__init__()
        self.set_shader('simple')
        self.set_primitive_type('lines')

    def set_data(self, bounds=NDC, color=None):
        # Set the position.
        x0, y0, x1, y1 = bounds
        arr = np.array([[x0, y0],
                        [x0, y1],
                        [x0, y1],
                        [x1, y1],
                        [x1, y1],
                        [x1, y0],
                        [x1, y0],
                        [x0, y0]], dtype=np.float32)
        self.program['a_position'] = self.transforms.apply(arr)

        # Set the color
        self.program['u_color'] = _get_color(color, self._default_color)


class AxesVisual(BaseVisual):
    _default_color = (.2, .2, .2, 1.)

    def __init__(self):
        super(AxesVisual, self).__init__()
        self.set_shader('simple')
        self.set_primitive_type('lines')

    def set_data(self, xs=(), ys=(), bounds=NDC, color=None):
        # Set the position.
        arr = [[x, bounds[1], x, bounds[3]] for x in xs]
        arr += [[bounds[0], y, bounds[2], y] for y in ys]
        arr = np.hstack(arr or [[]]).astype(np.float32)
        arr = arr.reshape((-1, 2)).astype(np.float32)
        position = self.transforms.apply(arr)
        self.program['a_position'] = position

        # Set the color
        self.program['u_color'] = _get_color(color, self._default_color)
