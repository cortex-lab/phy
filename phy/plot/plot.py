# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np

from phy.io.array import _accumulate
from .base import BaseCanvas
from .interact import Grid, Boxed, Stacked
from .panzoom import PanZoom
from .transform import NDC
from .utils import _get_array
from .visuals import ScatterVisual, PlotVisual, HistogramVisual, LineVisual


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

# NOTE: we ensure that we only create every type *once*, so that
# View._items has only one key for any class.
_SCATTER_CLASSES = {}


def _make_scatter_class(marker):
    """Return a temporary ScatterVisual class with a given marker."""
    name = 'ScatterVisual' + marker.title()
    if name not in _SCATTER_CLASSES:
        cls = type(name, (ScatterVisual,), {'_default_marker': marker})
        _SCATTER_CLASSES[name] = cls
    return _SCATTER_CLASSES[name]


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class View(BaseCanvas):
    """High-level plotting canvas."""
    _default_box_index = (0,)

    def __init__(self, layout=None, shape=None, n_plots=None, origin=None,
                 box_bounds=None, box_pos=None, box_size=None, **kwargs):
        if not kwargs.get('keys', None):
            kwargs['keys'] = None
        super(View, self).__init__(**kwargs)

        if layout == 'grid':
            self._default_box_index = (0, 0)
            self.grid = Grid(shape)
            self.grid.attach(self)

        elif layout == 'boxed':
            self.n_plots = (len(box_bounds)
                            if box_bounds is not None else len(box_pos))
            self.boxed = Boxed(box_bounds=box_bounds,
                               box_pos=box_pos,
                               box_size=box_size)
            self.boxed.attach(self)

        elif layout == 'stacked':
            self.n_plots = n_plots
            self.stacked = Stacked(n_plots, margin=.1, origin=origin)
            self.stacked.attach(self)

        self.panzoom = PanZoom(aspect=None, constrain_bounds=NDC)
        self.panzoom.attach(self)

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
