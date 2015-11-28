# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict, OrderedDict

import numpy as np

from .base import BaseCanvas
from .interact import Grid, Boxed, Stacked
from .panzoom import PanZoom
from .transform import NDC
from .utils import _get_array
from .visuals import ScatterVisual, PlotVisual, HistogramVisual


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class Accumulator(object):
    """Accumulate arrays for concatenation."""
    def __init__(self):
        self._data = defaultdict(list)

    def add(self, name, val):
        self._data[name].append(val)

    def __getitem__(self, name):
        return np.vstack(self._data[name]).astype(np.float32)


def _accumulate(data_list):
    acc = Accumulator()
    names = set()
    for data in data_list:
        for name, val in data.items():
            names.add(name)
            acc.add(name, val)
    return {name: acc[name] for name in names}


def _make_scatter_class(marker):
    return type('ScatterVisual' + marker.title(),
                (ScatterVisual,), {'_default_marker': marker})


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class BaseView(BaseCanvas):
    """High-level plotting canvas."""
    _default_box_index = (0,)

    def __init__(self, **kwargs):
        if not kwargs.get('keys', None):
            kwargs['keys'] = 'interactive'
        super(BaseView, self).__init__(**kwargs)
        self.clear()

    def clear(self):
        self._items = OrderedDict()

    def _add_item(self, cls, *args, **kwargs):
        data = cls.validate(*args, **kwargs)
        n = cls.vertex_count(**data)
        box_index = kwargs.get('box_index', self._default_box_index)
        k = len(box_index) if hasattr(box_index, '__len__') else 1
        box_index = _get_array(box_index, (n, k))
        data['box_index'] = box_index
        if cls not in self._items:
            self._items[cls] = []
        self._items[cls].append(data)
        return data

    def plot(self, *args, **kwargs):
        return self._add_item(PlotVisual, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        cls = _make_scatter_class(kwargs.pop('marker',
                                             ScatterVisual._default_marker))
        return self._add_item(cls, *args, **kwargs)

    def hist(self, *args, **kwargs):
        return self._add_item(HistogramVisual, *args, **kwargs)

    def __getitem__(self, box_index):
        self._default_box_index = box_index
        return self

    def build(self):
        for cls, data_list in self._items.items():
            data = _accumulate(data_list)
            box_index = data.pop('box_index')
            visual = cls()
            self.add_visual(visual)
            visual.set_data(**data)
            visual.program['a_box_index'] = box_index


class GridView(BaseView):
    """A 2D grid with clipping."""
    _default_box_index = (0, 0)

    def __init__(self, shape=None, **kwargs):
        super(GridView, self).__init__(**kwargs)

        self.grid = Grid(shape)
        self.grid.attach(self)

        self.panzoom = PanZoom(aspect=None, constrain_bounds=NDC)
        self.panzoom.attach(self)


class BoxedView(BaseView):
    """Subplots at arbitrary positions"""
    def __init__(self, box_bounds, **kwargs):
        super(BoxedView, self).__init__(**kwargs)
        self.n_plots = len(box_bounds)

        self.boxed = Boxed(box_bounds)
        self.boxed.attach(self)

        self.panzoom = PanZoom(aspect=None, constrain_bounds=NDC)
        self.panzoom.attach(self)


class StackedView(BaseView):
    """Stacked subplots"""
    def __init__(self, n_plots, **kwargs):
        super(StackedView, self).__init__(**kwargs)
        self.n_plots = n_plots

        self.stacked = Stacked(n_plots, margin=.1)
        self.stacked.attach(self)

        self.panzoom = PanZoom(aspect=None, constrain_bounds=NDC)
        self.panzoom.attach(self)
