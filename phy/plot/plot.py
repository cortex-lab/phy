# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict, OrderedDict
from contextlib import contextmanager

import numpy as np

from .base import BaseCanvas
from .interact import Grid, Boxed, Stacked
from .panzoom import PanZoom
from .transform import NDC
from .utils import _get_array
from .visuals import ScatterVisual, PlotVisual, HistogramVisual, LineVisual


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _flatten(l):
    return [item for sublist in l for item in sublist]


class Accumulator(object):
    """Accumulate arrays for concatenation."""
    def __init__(self):
        self._data = defaultdict(list)

    def add(self, name, val):
        """Add an array."""
        self._data[name].append(val)

    def get(self, name):
        """Return the list of arrays for a given name."""
        return _flatten(self._data[name])

    @property
    def names(self):
        """List of names."""
        return set(self._data)

    def __getitem__(self, name):
        """Concatenate all arrays with a given name."""
        return np.vstack(self._data[name]).astype(np.float32)


def _accumulate(data_list, no_concat=()):
    """Concatenate a list of dicts `(name, array)`.

    You can specify some names which arrays should not be concatenated.
    This is necessary with lists of plots with different sizes.

    """
    acc = Accumulator()
    for data in data_list:
        for name, val in data.items():
            acc.add(name, val)
    out = {name: acc[name] for name in acc.names if name not in no_concat}

    # Some variables should not be concatenated but should be kept as lists.
    # This is when there can be several arrays of variable length (NumPy
    # doesn't support ragged arrays).
    out.update({name: acc.get(name) for name in no_concat})
    return out


def _make_scatter_class(marker):
    """Return a temporary ScatterVisual class with a given marker."""
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
            kwargs['keys'] = None
        super(BaseView, self).__init__(**kwargs)
        self.clear()

    def clear(self):
        """Reset the view."""
        self._items = OrderedDict()
        self.visuals = []

    def _add_item(self, cls, *args, **kwargs):
        """Add a plot item."""
        box_index = kwargs.pop('box_index', self._default_box_index)

        data = cls.validate(*args, **kwargs)
        n = cls.vertex_count(**data)

        if not isinstance(box_index, np.ndarray):
            k = len(box_index) if hasattr(box_index, '__len__') else 1
            box_index = _get_array(box_index, (n, k))
        data['box_index'] = box_index

        if cls not in self._items:
            self._items[cls] = []
        self._items[cls].append(data)
        return data

    def plot(self, *args, **kwargs):
        """Add a line plot."""
        return self._add_item(PlotVisual, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        """Add a scatter plot."""
        cls = _make_scatter_class(kwargs.pop('marker',
                                             ScatterVisual._default_marker))
        return self._add_item(cls, *args, **kwargs)

    def hist(self, *args, **kwargs):
        """Add some histograms."""
        return self._add_item(HistogramVisual, *args, **kwargs)

    def lines(self, *args, **kwargs):
        """Add some lines."""
        return self._add_item(LineVisual, *args, **kwargs)

    def __getitem__(self, box_index):
        self._default_box_index = box_index
        return self

    def build(self):
        """Build all added items.

        Visuals are created, added, and built. The `set_data()` methods can
        be called afterwards.

        """
        for cls, data_list in self._items.items():
            # Some variables are not concatenated. They are specified
            # in `allow_list`.
            data = _accumulate(data_list, cls.allow_list)
            box_index = data.pop('box_index')
            visual = cls()
            self.add_visual(visual)
            visual.set_data(**data)
            # NOTE: visual.program.__contains__ is implemented in vispy master
            # so we can replace this with `if 'a_box_index' in visual.program`
            # after the next VisPy release.
            if 'a_box_index' in visual.program._code_variables:
                visual.program['a_box_index'] = box_index
        self.update()

    @contextmanager
    def building(self):
        """Context manager to specify the plots."""
        self.clear()
        yield
        self.build()


class SimpleView(BaseView):
    """A simple view."""
    def __init__(self, shape=None, **kwargs):
        super(SimpleView, self).__init__(**kwargs)

        self.panzoom = PanZoom(aspect=None, constrain_bounds=NDC)
        self.panzoom.attach(self)


class GridView(BaseView):
    """A 2D grid with clipping."""
    _default_box_index = (0, 0)

    def __init__(self, shape=None, **kwargs):
        super(GridView, self).__init__(**kwargs)

        self.grid = Grid(shape)
        self.grid.attach(self)

        self.panzoom = PanZoom(aspect=None, constrain_bounds=NDC)
        self.panzoom.attach(self)

    def build(self):
        n, m = self.grid.shape
        a = .045  # margin
        for i in range(n):
            for j in range(m):
                self[i, j].lines(x0=[-1, +1, +1, -1],
                                 y0=[-1, -1, +1, +1],
                                 x1=[+1, +1, -1, -1],
                                 y1=[-1, +1, +1, -1],
                                 data_bounds=[-1 + a, -1 + a, 1 - a, 1 - a],
                                 )
        super(GridView, self).build()


class BoxedView(BaseView):
    """Subplots at arbitrary positions"""
    def __init__(self, box_bounds=None, box_pos=None, box_size=None, **kwargs):
        super(BoxedView, self).__init__(**kwargs)
        self.n_plots = (len(box_bounds)
                        if box_bounds is not None else len(box_pos))

        self.boxed = Boxed(box_bounds=box_bounds,
                           box_pos=box_pos,
                           box_size=box_size)
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
