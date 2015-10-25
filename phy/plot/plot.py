# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import groupby
from collections import defaultdict

import numpy as np

from .base import BaseCanvas
from .interact import Grid  # Boxed, Stacked
from .visuals import _get_array, ScatterVisual, PlotVisual, HistogramVisual


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class Accumulator(object):
    """Accumulate arrays for concatenation."""
    def __init__(self):
        self._size = defaultdict(int)
        self._data = defaultdict(list)

    @property
    def size(self):
        return self._size[list(self._size.keys())[0]]

    @property
    def data(self):
        return {name: self[name] for name in self._data}

    def __setitem__(self, name, val):
        self._size[name] += len(val)
        self._data[name].append(val)

    def __getitem__(self, name):
        size = self.size
        assert all(s == size for s in self._size.values())
        return np.vstack(self._data[name]).astype(np.float32)


#------------------------------------------------------------------------------
# Base plotting interface
#------------------------------------------------------------------------------

class SubView(object):
    def __init__(self, idx):
        self.spec = {'idx': idx}

    @property
    def visual_class(self):
        return self.spec.get('visual_class', None)

    def _set(self, visual_class, loc):
        self.spec['visual_class'] = visual_class
        self.spec.update(loc)

    def __getattr__(self, name):
        return self.spec[name]

    def scatter(self, x, y, color=None, size=None, marker=None):
        # Validate x and y.
        assert x.ndim == y.ndim == 1
        assert x.shape == y.shape
        n = x.shape[0]
        # Set the color and size.
        color = _get_array(color, (n, 4), ScatterVisual._default_color)
        size = _get_array(size, (n, 1), ScatterVisual._default_marker_size)
        # Default marker.
        marker = marker or ScatterVisual._default_marker
        # Set the spec.
        loc = dict(x=x, y=y, color=color, size=size, marker=marker)
        return self._set(ScatterVisual, loc)

    def plot(self, x, y, color=None):
        loc = locals()
        return self._set(PlotVisual, loc)

    def hist(self, hist, color=None):
        loc = locals()
        return self._set(HistogramVisual, loc)

    def __repr__(self):
        return str(self.spec)


class BaseView(BaseCanvas):
    def __init__(self, interacts):
        super(BaseView, self).__init__()
        # Attach the passed interacts to the current canvas.
        for interact in interacts:
            interact.attach(self)
        self.subviews = {}

    # To override
    # -------------------------------------------------------------------------

    def get_box_ndim(self):
        raise NotImplementedError()

    def iter_index(self):
        raise NotImplementedError()

    # Internal methods
    # -------------------------------------------------------------------------

    def iter_subviews(self):
        for idx in self.iter_index():
            sv = self.subviews.get(idx, None)
            if sv:
                yield sv

    def __getitem__(self, idx):
        sv = SubView(idx)
        self.subviews[idx] = sv
        return sv

    def _build_scatter(self, subviews, marker):
        """Build all scatter subviews with the same marker type."""

        ac = Accumulator()
        for sv in subviews:
            assert sv.marker == marker
            n = len(sv.x)
            ac['pos'] = np.c_[sv.x, sv.y]
            ac['color'] = sv.color
            ac['size'] = sv.size
            ac['box_index'] = np.tile(sv.idx, (n, 1))

        v = ScatterVisual(marker=marker)
        v.attach(self)
        v.set_data(pos=ac['pos'], color=ac['color'], size=ac['size'])
        v.program['a_box_index'] = ac['box_index']

    def build(self):
        """Build all visuals."""
        for visual_class, subviews in groupby(self.iter_subviews(),
                                              lambda sv: sv.visual_class):
            if visual_class == ScatterVisual:
                for marker, subviews_scatter in groupby(subviews,
                                                        lambda sv: sv.marker):
                    self._build_scatter(subviews_scatter, marker)
            elif visual_class == PlotVisual:
                self._build_plot(subviews)
            elif visual_class == HistogramVisual:
                self._build_histogram(subviews)


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class GridView(BaseView):
    def __init__(self, n_rows, n_cols):
        self.n_rows, self.n_cols = n_rows, n_cols
        interacts = [Grid(n_rows, n_cols)]
        super(GridView, self).__init__(interacts)

    def get_box_ndim(self):
        return 2

    def iter_index(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                yield (i, j)


class StackedView(BaseView):
    def __init__(self, n_plots):
        super(StackedView, self).__init__()


class BoxedView(BaseView):
    def __init__(self, box_positions):
        super(BoxedView, self).__init__()
