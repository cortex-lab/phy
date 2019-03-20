# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np

from .axes import Axes
from .base import BaseCanvas
from .interact import Grid, Boxed, Stacked, Lasso
from .panzoom import PanZoom
from phy.plot.utils import _get_array
from phy.utils._types import _as_tuple

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class PlotCanvas(BaseCanvas):
    """Plotting canvas."""
    _default_box_index = (0,)

    def __init__(self, layout=None, shape=None, n_plots=None, origin=None,
                 box_bounds=None, box_pos=None, box_size=None,
                 with_panzoom=True, with_lasso=False, with_axes=False,
                 **kwargs):
        super(PlotCanvas, self).__init__(**kwargs)
        self.layout = layout

        # Constrain pan zoom.
        self.constrain_bounds = [-2, -2, +2, +2]
        if layout == 'grid':
            self._default_box_index = (0, 0)
            self.grid = Grid(shape)
            self.grid.attach(self)
            self.interact = self.grid
            self.constrain_bounds = [-1, -1, +1, +1]

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

        if with_panzoom:
            self.enable_panzoom()
        if with_lasso:
            self.enable_lasso()
        if with_axes:
            self.enable_axes()

        if layout == 'grid':
            self.interact.add_boxes(self)

    def add_visual(self, visual, box_index=None):
        if self.interact:
            @self.on_next_paint
            def set_box_index():
                visual.program['a_box_index'] = self.make_box_index(visual, box_index=box_index)
        super(PlotCanvas, self).add_visual(visual)

    def __getitem__(self, box_index):
        self._default_box_index = _as_tuple(box_index)
        return self

    def add(self, visual, *args, **kwargs):
        self.add_visual(visual, box_index=self._default_box_index)
        visual.set_data(*args, **kwargs)

    def make_box_index(self, visual, box_index=None):
        if box_index is None:
            box_index = self._default_box_index
        if not isinstance(box_index, np.ndarray):
            n = visual.n_vertices
            k = len(self._default_box_index)
            box_index = _get_array(box_index, (n, k))
        return box_index

    def enable_panzoom(self):
        self.panzoom = PanZoom(aspect=None, constrain_bounds=self.constrain_bounds)
        self.panzoom.attach(self)

    def enable_lasso(self):
        self.lasso = Lasso()
        self.lasso.attach(self)

    def enable_axes(self, data_bounds=None):
        self.axes = Axes(data_bounds=data_bounds)
        self.axes.attach(self)
