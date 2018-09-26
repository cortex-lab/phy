# -*- coding: utf-8 -*-

"""Grid and axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from math import log2

import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoLocator, MaxNLocator


from .transform import Range, NDC
from .visuals import LineVisual, TextVisual
from phy.utils import Bunch
from phy import connect


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class AxisLocator(object):
    def __init__(self, nbins=24):
        self.locator = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
        self.formatter = ScalarFormatter()

    def get_ticks(self, bounds):
        x0, y0, x1, y1 = bounds
        dx = x1 - x0
        dy = y1 - y0
        xticks = self.locator.tick_values(x0 - dx, x1 + dx)
        yticks = self.locator.tick_values(y0 - dy, y1 + dy)
        return xticks, yticks

    def format(self, val):
        return self.formatter.format_data(val).replace('−', '-')


#------------------------------------------------------------------------------
# Grid visual
#------------------------------------------------------------------------------

def _fix_coordinate_in_visual(visual, coord):
    assert coord in ('x', 'y')
    visual.inserter.insert_vert(
        'gl_Position.{coord} = pos_orig.{coord};'.format(coord=coord),
        'after_transforms')


def _set_line_data(xticks, yticks):
    xdata = np.c_[xticks, -np.ones(len(xticks)), xticks, np.ones(len(xticks))]
    ydata = np.c_[-np.ones(len(yticks)), yticks, np.ones(len(yticks)), yticks]
    return xdata, ydata


class Grid(object):
    default_color = (1, 1, 1, .5)

    def __init__(self, color=None):
        self.locator = AxisLocator()
        self.color = color or self.default_color

        self.xvisual = LineVisual()
        self.yvisual = LineVisual()
        self.txvisual = TextVisual()
        self.tyvisual = TextVisual()

        _fix_coordinate_in_visual(self.xvisual, 'y')
        _fix_coordinate_in_visual(self.yvisual, 'x')
        _fix_coordinate_in_visual(self.txvisual, 'y')
        _fix_coordinate_in_visual(self.tyvisual, 'x')

        self._last_log_zoom = (1, 1)
        self._last_pan = (0, 0)

    def set_bounds(self, bounds):
        xticks, yticks = self.locator.get_ticks(bounds)
        xdata, ydata = _set_line_data(xticks, yticks)

        xpos, ypos = xdata[:, :2], ydata[:, 2:]

        xtext = [self.locator.format(_) for _ in xticks]
        ytext = [self.locator.format(_) for _ in yticks]

        self.xvisual.set_data(xdata, color=self.color)
        self.yvisual.set_data(ydata, color=self.color)

        self.txvisual.set_data(pos=xpos, text=xtext, anchor=(0, 1.05))
        self.tyvisual.set_data(pos=ypos, text=ytext, anchor=(-1.05, 0))

    def attach(self, canvas):
        canvas.add_visual(self.xvisual)
        canvas.add_visual(self.yvisual)
        canvas.add_visual(self.txvisual)
        canvas.add_visual(self.tyvisual)
        self.set_bounds(NDC)

        @connect(sender=canvas.panzoom)
        def on_zoom(sender, zoom):
            zx, zy = zoom
            ix, iy = int(log2(zx)), int(log2(zy))
            if (ix, iy) != self._last_log_zoom:
                self._last_log_zoom = ix, iy
                self.set_bounds(canvas.panzoom.get_range())

        @connect(sender=canvas.panzoom)
        def on_pan(sender, pan):
            px, py = pan
            zx, zy = canvas.panzoom.zoom
            tx, ty = int(px * zx), int(py * zy)
            if (tx, ty) != self._last_pan:
                self._last_pan = tx, ty
                self.set_bounds(canvas.panzoom.get_range())
