# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby

import numpy as np

from phy.utils import Bunch, _is_array_like
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

    def __init__(self, **kwargs):
        super(BaseView, self).__init__(**kwargs)
        self._default_box_index = None
        self.clear()

    def clear(self):
        self._items = defaultdict(list)

    def _add_item(self, cls, *args, **kwargs):
        data = cls.validate(*args, **kwargs)
        data['box_index'] = kwargs.get('box_index', self._default_box_index)
        self._items[cls].append(data)

    def plot(self, *args, **kwargs):
        self._add_item(PlotVisual, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        cls = _make_scatter_class(kwargs.get('marker',
                                             ScatterVisual._default_marker))
        self._add_item(cls, *args, **kwargs)

    def hist(self, *args, **kwargs):
        self._add_item(HistogramVisual, *args, **kwargs)

    def __getitem__(self, box_index):

        @contextmanager
        def box_index_ctx():
            self._default_box_index = box_index
            yield
            self._default_box_index = None

        with box_index_ctx():
            return self

    def build(self):
        for cls, data_list in self._items.items():
            data = _accumulate(data_list)
            box_index = data.pop('box_index')
            visual = cls()
            self.add_visual(visual)
            visual.set_data(**data)
            try:
                visual.program['a_box_index']
                visual.program['a_box_index'] = box_index
            except KeyError:
                pass


# class GridView(BaseView):
#     """A 2D grid with clipping."""
#     def __init__(self, shape, **kwargs):
#         self.n_rows, self.n_cols = shape
#         pz = PanZoom(aspect=None, constrain_bounds=NDC)
#         interacts = [Grid(shape), pz]
#         super(GridView, self).__init__(interacts, **kwargs)


# class BoxedView(BaseView):
#     """Subplots at arbitrary positions"""
#     def __init__(self, box_bounds, **kwargs):
#         self.n_plots = len(box_bounds)
#         self._boxed = Boxed(box_bounds)
#         self._pz = PanZoom(aspect=None, constrain_bounds=NDC)
#         interacts = [self._boxed, self._pz]
#         super(BoxedView, self).__init__(interacts, **kwargs)


# class StackedView(BaseView):
#     """Stacked subplots"""
#     def __init__(self, n_plots, **kwargs):
#         self.n_plots = n_plots
#         pz = PanZoom(aspect=None, constrain_bounds=NDC)
#         interacts = [Stacked(n_plots, margin=.1), pz]
#         super(StackedView, self).__init__(interacts, **kwargs)
