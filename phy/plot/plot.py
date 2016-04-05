# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import logging

import numpy as np

from phy.io.array import _accumulate, _in_polygon
from phy.utils._types import _as_tuple
from .base import BaseCanvas
from .interact import Grid, Boxed, Stacked
from .panzoom import PanZoom
from .utils import _get_array
from .visuals import (ScatterVisual, PlotVisual, HistogramVisual,
                      LineVisual, TextVisual, PolygonVisual,
                      UniformScatterVisual, UniformPlotVisual,
                      )

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

# NOTE: we ensure that we only create every type *once*, so that
# View._items has only one key for any class.
_CLASSES = {}


def _hash(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(s).hexdigest()[:8]


def _make_class(cls, **kwargs):
    """Return a custom Visual class with given parameters."""
    kwargs = {k: (v if v is not None else getattr(cls, k, None))
              for k, v in kwargs.items()}
    # The class name contains a hash of the custom parameters.
    name = cls.__name__ + '_' + _hash(kwargs)
    if name not in _CLASSES:
        logger.log(5, "Create class %s %s.", name, kwargs)
        cls = type(name, (cls,), kwargs)
        _CLASSES[name] = cls
    return _CLASSES[name]


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class View(BaseCanvas):
    """High-level plotting canvas."""
    _default_box_index = (0,)

    def __init__(self, layout=None, shape=None, n_plots=None, origin=None,
                 box_bounds=None, box_pos=None, box_size=None,
                 enable_lasso=False,
                 **kwargs):
        if not kwargs.get('keys', None):
            kwargs['keys'] = None
        super(View, self).__init__(**kwargs)
        self.layout = layout

        if layout == 'grid':
            self._default_box_index = (0, 0)
            self.grid = Grid(shape)
            self.grid.attach(self)
            self.interact = self.grid

        elif layout == 'boxed':
            self.n_plots = (len(box_bounds)
                            if box_bounds is not None else len(box_pos))
            self.boxed = Boxed(box_bounds=box_bounds,
                               box_pos=box_pos,
                               box_size=box_size)
            self.boxed.attach(self)
            self.interact = self.boxed

        elif layout == 'stacked':
            self.n_plots = n_plots
            self.stacked = Stacked(n_plots, margin=.1, origin=origin)
            self.stacked.attach(self)
            self.interact = self.stacked

        else:
            self.interact = None

        self.panzoom = PanZoom(aspect=None,
                               constrain_bounds=[-2, -2, +2, +2])
        self.panzoom.attach(self)

        if enable_lasso:
            self.lasso = Lasso()
            self.lasso.attach(self)
        else:
            self.lasso = None

        self.clear()

    def clear(self):
        """Reset the view."""
        self._items = OrderedDict()
        self.visuals = []
        self.update()

    def _add_item(self, cls, *args, **kwargs):
        """Add a plot item."""
        box_index = kwargs.pop('box_index', self._default_box_index)

        data = cls.validate(*args, **kwargs)
        n = cls.vertex_count(**data)

        if not isinstance(box_index, np.ndarray):
            k = len(self._default_box_index)
            box_index = _get_array(box_index, (n, k))
        data['box_index'] = box_index

        if cls not in self._items:
            self._items[cls] = []
        self._items[cls].append(data)
        return data

    def _plot_uniform(self, *args, **kwargs):
        cls = _make_class(UniformPlotVisual,
                          _default_color=kwargs.pop('color', None),
                          )
        return self._add_item(cls, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """Add a line plot."""
        if kwargs.pop('uniform', None):
            return self._plot_uniform(*args, **kwargs)
        return self._add_item(PlotVisual, *args, **kwargs)

    def _scatter_uniform(self, *args, **kwargs):
        cls = _make_class(UniformScatterVisual,
                          _default_marker=kwargs.pop('marker', None),
                          _default_marker_size=kwargs.pop('size', None),
                          _default_color=kwargs.pop('color', None),
                          )
        return self._add_item(cls, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        """Add a scatter plot."""
        if kwargs.pop('uniform', None):
            return self._scatter_uniform(*args, **kwargs)
        cls = _make_class(ScatterVisual,
                          _default_marker=kwargs.pop('marker', None),
                          )
        return self._add_item(cls, *args, **kwargs)

    def hist(self, *args, **kwargs):
        """Add some histograms."""
        return self._add_item(HistogramVisual, *args, **kwargs)

    def text(self, *args, **kwargs):
        """Add text."""
        return self._add_item(TextVisual, *args, **kwargs)

    def lines(self, *args, **kwargs):
        """Add some lines."""
        return self._add_item(LineVisual, *args, **kwargs)

    def __getitem__(self, box_index):
        self._default_box_index = _as_tuple(box_index)
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
                visual.program['a_box_index'] = box_index.astype(np.float32)
        # TODO: refactor this when there is the possibility to update existing
        # visuals without recreating the whole scene.
        if self.lasso:
            self.lasso.create_visual()
        self.update()

    def get_pos_from_mouse(self, pos, box):
        # From window coordinates to NDC (pan & zoom taken into account).
        pos = self.panzoom.get_mouse_pos(pos)
        # From NDC to data coordinates.
        pos = self.interact.imap(pos, box) if self.interact else pos
        return pos

    @contextmanager
    def building(self):
        """Context manager to specify the plots."""
        self.clear()
        yield
        self.build()


#------------------------------------------------------------------------------
# Interactive tools
#------------------------------------------------------------------------------

class Lasso(object):
    def __init__(self):
        self._points = []
        self.view = None
        self.visual = None
        self.box = None

    def add(self, pos):
        self._points.append(pos)
        self.update_visual()

    @property
    def polygon(self):
        l = self._points
        # Close the polygon.
        # l = l + l[0] if len(l) else l
        out = np.array(l, dtype=np.float64)
        out = np.reshape(out, (out.size // 2, 2))
        assert out.ndim == 2
        assert out.shape[1] == 2
        return out

    def clear(self):
        self._points = []
        self.box = None
        self.update_visual()

    @property
    def count(self):
        return len(self._points)

    def in_polygon(self, pos):
        return _in_polygon(pos, self.polygon)

    def attach(self, view):
        view.connect(self.on_mouse_press)
        self.view = view

    def create_visual(self):
        self.visual = PolygonVisual()
        self.view.add_visual(self.visual)
        self.update_visual()

    def update_visual(self):
        if not self.visual:
            return
        # Update the polygon.
        self.visual.set_data(pos=self.polygon)
        # Set the box index for the polygon, depending on the box
        # where the first point was clicked in.
        box = (self.box if self.box is not None
               else self.view._default_box_index)
        k = len(self.view._default_box_index)
        n = self.visual.vertex_count(pos=self.polygon)
        box_index = _get_array(box, (n, k)).astype(np.float32)
        self.visual.program['a_box_index'] = box_index
        self.view.update()

    def on_mouse_press(self, e):
        if 'Control' in e.modifiers:
            if e.button == 1:
                # Find the box.
                ndc = self.view.panzoom.get_mouse_pos(e.pos)
                # NOTE: we don't update the box after the second point.
                # In other words, the first point determines the box for the
                # lasso.
                if self.box is None and self.view.interact:
                    self.box = self.view.interact.get_closest_box(ndc)
                # Transform from window coordinates to NDC.
                pos = self.view.get_pos_from_mouse(e.pos, self.box)
                self.add(pos)
            else:
                self.clear()
                self.box = None
