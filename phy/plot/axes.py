# -*- coding: utf-8 -*-

"""Axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from math import log2

import numpy as np
from matplotlib.ticker import MaxNLocator


from .transform import NDC, Range
from .visuals import LineVisual, TextVisual
from phy import connect


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class AxisLocator(object):
    def __init__(self, nbins=24, data_bounds=None):
        self.locator = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
        self.data_bounds = data_bounds
        self._tr = Range(from_bounds=NDC, to_bounds=self.data_bounds)
        self._tri = self._tr.inverse()

    def _transform_ticks(self, xticks, yticks):
        """From data coordinates to view coordinates."""
        nx, ny = len(xticks), len(yticks)
        arr = np.zeros((nx + ny, 2))
        arr[:nx, 0] = xticks
        arr[nx:, 1] = yticks
        out = self._tri.apply(arr)
        return out[:nx, 0], out[nx:, 1]

    def set_view_bounds(self, view_bounds):
        x0, y0, x1, y1 = view_bounds
        dx = x1 - x0
        dy = y1 - y0

        # Get the bounds in data coordinates.
        ((dx0, dy0), (dx1, dy1)) = self._tr.apply([
            [x0 - dx, y0 - dy],
            [x1 + dx, y1 + dy]])

        # Compute the ticks in data coordinates.
        self.xticks = self.locator.tick_values(dx0, dx1)
        self.yticks = self.locator.tick_values(dy0, dy1)

        # Get the ticks in view coordinates.
        self.xticks_view, self.yticks_view = self._transform_ticks(self.xticks, self.yticks)

        # Get the text in data coordinates.
        self.xtext = ['%g' % v for v in self.xticks]
        self.ytext = ['%g' % v for v in self.yticks]


#------------------------------------------------------------------------------
# Axes visual
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


class Axes(object):
    default_color = (1, 1, 1, .25)

    def __init__(self, color=None, data_bounds=None):
        self.locator = AxisLocator(data_bounds=data_bounds)
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
        self.locator.set_view_bounds(bounds)
        # Get the text data.
        xtext, ytext = self.locator.xtext, self.locator.ytext
        # GPU data for the grid.
        xdata, ydata = _set_line_data(self.locator.xticks_view, self.locator.yticks_view)
        # Position of the text in view coordinates.
        xpos, ypos = xdata[:, :2], ydata[:, 2:]

        # Set the visuals data.
        self.xvisual.set_data(xdata, color=self.color)
        self.yvisual.set_data(ydata, color=self.color)

        self.txvisual.set_data(pos=xpos, text=xtext, anchor=(0, 1.02))
        self.tyvisual.set_data(pos=ypos, text=ytext, anchor=(-1.02, 0))

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
