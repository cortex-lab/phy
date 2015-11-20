# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import groupby
from collections import defaultdict

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

    def __setitem__(self, name, val):
        self._data[name].append(val)

    def __getitem__(self, name):
        return np.vstack(self._data[name]).astype(np.float32)


#------------------------------------------------------------------------------
# Base plotting interface
#------------------------------------------------------------------------------

def _prepare_scatter(x, y, color=None, size=None, marker=None):
    x = np.asarray(x)
    y = np.asarray(y)
    # Validate x and y.
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    n = x.shape[0]
    # Set the color and size.
    color = _get_array(color, (n, 4), ScatterVisual._default_color)
    size = _get_array(size, (n, 1), ScatterVisual._default_marker_size)
    # Default marker.
    marker = marker or ScatterVisual._default_marker
    return dict(x=x, y=y, color=color, size=size, marker=marker)


def _prepare_plot(x, y, color=None, depth=None, data_bounds=None):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    # Validate x and y.
    assert x.ndim == y.ndim == 2
    assert x.shape == y.shape
    n_plots, n_samples = x.shape
    # Get the colors.
    color = _get_array(color, (n_plots, 4), PlotVisual._default_color)
    # Get the depth.
    depth = _get_array(depth, (n_plots, 1), 0)
    return dict(x=x, y=y, color=color, depth=depth, data_bounds=data_bounds)


def _prepare_hist(data, color=None):
    # Validate data.
    if data.ndim == 1:
        data = data[np.newaxis, :]
    assert data.ndim == 2
    n_hists, n_samples = data.shape
    # Get the colors.
    color = _get_array(color, (n_hists, 4), HistogramVisual._default_color)
    return dict(data=data, color=color)


def _prepare_box_index(box_index, n):
    if not _is_array_like(box_index):
        box_index = np.tile(box_index, (n, 1))
    box_index = np.asarray(box_index, dtype=np.int32)
    assert box_index.ndim == 2
    assert box_index.shape[0] == n
    return box_index


def _build_scatter(items):
    """Build scatter items and return parameters for `set_data()`."""

    ac = Accumulator()
    for item in items:
        # The item data has already been prepared.
        n = len(item.data.x)
        ac['pos'] = np.c_[item.data.x, item.data.y]
        ac['color'] = item.data.color
        ac['size'] = item.data.size
        ac['box_index'] = _prepare_box_index(item.box_index, n)

    return (dict(pos=ac['pos'], color=ac['color'], size=ac['size']),
            ac['box_index'])


def _build_plot(items):
    """Build all plot items and return parameters for `set_data()`."""

    ac = Accumulator()
    for item in items:
        n = item.data.x.size
        ac['x'] = item.data.x
        ac['y'] = item.data.y
        ac['depth'] = item.data.depth
        ac['plot_colors'] = item.data.color
        ac['box_index'] = _prepare_box_index(item.box_index, n)

    return (dict(x=ac['x'], y=ac['y'],
                 plot_colors=ac['plot_colors'],
                 depth=ac['depth'],
                 data_bounds=item.data.data_bounds,
                 ),
            ac['box_index'])


def _build_histogram(items):
    """Build all histogram items and return parameters for `set_data()`."""

    ac = Accumulator()
    for item in items:
        n = item.data.data.size
        ac['data'] = item.data.data
        ac['hist_colors'] = item.data.color
        # NOTE: the `6 * ` comes from the histogram tesselation.
        ac['box_index'] = _prepare_box_index(item.box_index, 6 * n)

    return (dict(hist=ac['data'], hist_colors=ac['hist_colors']),
            ac['box_index'])


class ViewItem(Bunch):
    """A visual item that will be rendered in batch with other view items
    of the same type."""
    def __init__(self, base, visual_class=None, data=None, box_index=None):
        super(ViewItem, self).__init__(visual_class=visual_class,
                                       data=Bunch(data),
                                       box_index=box_index,
                                       to_build=True,
                                       )
        self._base = base

    def set_data(self, **kwargs):
        self.data.update(kwargs)
        self.to_build = True


class BaseView(BaseCanvas):
    """High-level plotting canvas."""

    def __init__(self, interacts, **kwargs):
        super(BaseView, self).__init__(**kwargs)
        # Attach the passed interacts to the current canvas.
        for interact in interacts:
            interact.attach(self)
        self._items = []  # List of view items instances.
        self._visuals = {}

    @property
    def panzoom(self):
        """PanZoom instance from the interact list, if it exists."""
        for interact in self.interacts:
            if isinstance(interact, PanZoom):
                return interact

    # To override
    # -------------------------------------------------------------------------

    def __getitem__(self, idx):
        class _Proxy(object):
            def scatter(s, *args, **kwargs):
                kwargs['box_index'] = idx
                return self.scatter(*args, **kwargs)

            def plot(s, *args, **kwargs):
                kwargs['box_index'] = idx
                return self.plot(*args, **kwargs)

            def hist(s, *args, **kwargs):
                kwargs['box_index'] = idx
                return self.hist(*args, **kwargs)

        return _Proxy()

    def _iter_items(self):
        """Iterate over all view items."""
        for item in self._items:
            yield item

    def _visuals_to_build(self):
        """Return the set of visual classes that need to be rebuilt."""
        visual_classes = set()
        for item in self._items:
            if item.to_build:
                visual_classes.add(item.visual_class)
        return visual_classes

    def _get_visual(self, key):
        """Create or return a visual from its class or tuple (class, param)."""
        if key not in self._visuals:
            # Create the visual.
            if isinstance(key, tuple):
                # Case of the scatter plot, where the visual depends on the
                # marker.
                v = key[0](key[1])
            else:
                v = key()
            # Attach the visual to the view.
            v.attach(self)
            # Store the visual for reuse.
            self._visuals[key] = v
        return self._visuals[key]

    # Public methods
    # -------------------------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Add a line plot."""
        box_index = kwargs.pop('box_index', None)
        data = _prepare_plot(*args, **kwargs)
        item = ViewItem(self, visual_class=PlotVisual,
                        data=data, box_index=box_index)
        self._items.append(item)
        return item

    def scatter(self, *args, **kwargs):
        """Add a scatter plot."""
        box_index = kwargs.pop('box_index', None)
        data = _prepare_scatter(*args, **kwargs)
        item = ViewItem(self, visual_class=ScatterVisual,
                        data=data, box_index=box_index)
        self._items.append(item)
        return item

    def hist(self, *args, **kwargs):
        """Add a histogram plot."""
        box_index = kwargs.pop('box_index', None)
        data = _prepare_hist(*args, **kwargs)
        item = ViewItem(self, visual_class=HistogramVisual,
                        data=data, box_index=box_index)
        self._items.append(item)
        return item

    def build(self):
        """Build all visuals."""
        visuals_to_build = self._visuals_to_build()

        for visual_class, items in groupby(self._iter_items(),
                                           lambda item: item.visual_class):
            items = list(items)

            # Skip visuals that do not need to be built.
            if visual_class not in visuals_to_build:
                continue

            # Histogram.
            # TODO: refactor this (DRY).
            if visual_class == HistogramVisual:
                data, box_index = _build_histogram(items)
                v = self._get_visual(HistogramVisual)
                v.set_data(**data)
                v.program['a_box_index'] = box_index
                for item in items:
                    item.to_build = False

            # Scatter.
            if visual_class == ScatterVisual:
                items_grouped = groupby(items, lambda item: item.data.marker)
                # One visual per marker type.
                for marker, items_scatter in items_grouped:
                    items_scatter = list(items_scatter)
                    data, box_index = _build_scatter(items_scatter)
                    v = self._get_visual((ScatterVisual, marker))
                    v.set_data(**data)
                    v.program['a_box_index'] = box_index
                    for item in items_scatter:
                        item.to_build = False

            # Plot.
            if visual_class == PlotVisual:
                items_grouped = groupby(items,
                                        lambda item: item.data.x.shape[1])
                # HACK: one visual per number of samples, because currently
                # a PlotVisual only accepts a regular (n_plots, n_samples)
                # array as input.
                for n_samples, items_plot in items_grouped:
                    items_plot = list(items_plot)
                    data, box_index = _build_plot(items_plot)
                    v = self._get_visual((PlotVisual, n_samples))
                    v.set_data(**data)
                    v.program['a_box_index'] = box_index
                    for item in items_plot:
                        item.to_build = False

        self.update()


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class GridView(BaseView):
    """A 2D grid with clipping."""
    def __init__(self, n_rows, n_cols, **kwargs):
        self.n_rows, self.n_cols = n_rows, n_cols
        pz = PanZoom(aspect=None, constrain_bounds=NDC)
        interacts = [Grid(n_rows, n_cols), pz]
        super(GridView, self).__init__(interacts, **kwargs)


class BoxedView(BaseView):
    """Subplots at arbitrary positions"""
    def __init__(self, box_bounds, **kwargs):
        self.n_plots = len(box_bounds)
        self._boxed = Boxed(box_bounds)
        self._pz = PanZoom(aspect=None, constrain_bounds=NDC)
        interacts = [self._boxed, self._pz]
        super(BoxedView, self).__init__(interacts, **kwargs)


class StackedView(BaseView):
    """Stacked subplots"""
    def __init__(self, n_plots, **kwargs):
        self.n_plots = n_plots
        pz = PanZoom(aspect=None, constrain_bounds=NDC)
        interacts = [Stacked(n_plots, margin=.1), pz]
        super(StackedView, self).__init__(interacts, **kwargs)
