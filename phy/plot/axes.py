# -*- coding: utf-8 -*-

"""Axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from matplotlib.ticker import MaxNLocator


from .transform import NDC, Range
from .visuals import LineVisual, TextVisual
from phylib import connect
from phylib.utils._types import _is_integer
from phy.gui.qt import _is_high_dpi


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class AxisLocator(object):
    _default_nbinsx = 24
    _default_nbinsy = 16
    # ticks are extended beyond the viewport for smooth transition
    # when panzooming: -2, -1, 0, +1, +2
    _bins_margin = 5
    _default_steps = (1, 2, 2.5, 5, 10)

    def __init__(self, nbinsx=None, nbinsy=None, data_bounds=None):
        """data_bounds is the initial bounds of the view in data coordinates."""
        self.data_bounds = data_bounds
        self._tr = Range(from_bounds=NDC, to_bounds=self.data_bounds)
        self._tri = self._tr.inverse()
        self.set_nbins(nbinsx, nbinsy)

    def set_nbins(self, nbinsx=None, nbinsy=None):
        nbinsx = self._bins_margin * nbinsx if _is_integer(nbinsx) else self._default_nbinsx
        nbinsy = self._bins_margin * nbinsy if _is_integer(nbinsy) else self._default_nbinsy
        self.locx = MaxNLocator(nbins=nbinsx, steps=self._default_steps)
        self.locy = MaxNLocator(nbins=nbinsy, steps=self._default_steps)

    def _transform_ticks(self, xticks, yticks):
        """From data coordinates to view coordinates."""
        nx, ny = len(xticks), len(yticks)
        arr = np.zeros((nx + ny, 2))
        arr[:nx, 0] = xticks
        arr[nx:, 1] = yticks
        out = self._tri.apply(arr)
        return out[:nx, 0], out[nx:, 1]

    def set_view_bounds(self, view_bounds=None):
        """Set the view bounds in NDC."""
        view_bounds = view_bounds or NDC
        x0, y0, x1, y1 = view_bounds
        dx = 2 * (x1 - x0)
        dy = 2 * (y1 - y0)

        # Get the bounds in data coordinates.
        ((dx0, dy0), (dx1, dy1)) = self._tr.apply([
            [x0 - dx, y0 - dy],
            [x1 + dx, y1 + dy]])

        # Compute the ticks in data coordinates.
        self.xticks = self.locx.tick_values(dx0, dx1)
        self.yticks = self.locy.tick_values(dy0, dy1)

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


def get_nbins(w, h):
    """Return a sensible number of bins on the x and y axes as a function
    of the window size."""
    if _is_high_dpi():  # pragma: no cover
        w, h = w // 2, h // 2
    return max(1, w // 150), max(1, h // 80)


def _quant_zoom(z):
    if z == 0:
        return 0  # pragma: no cover
    return int(z) if z >= 1 else -int(1. / z)


class Axes(object):
    default_color = (1, 1, 1, .25)

    def __init__(self, data_bounds=None, color=None, show_x=True, show_y=True):
        self.show_x = show_x
        self.show_y = show_y
        self.reset_data_bounds(data_bounds, do_update=False)
        self._create_visuals()
        self.color = color or self.default_color
        self._attached = None

    def reset_data_bounds(self, data_bounds, do_update=True):
        self.locator = AxisLocator(data_bounds=data_bounds)
        self.locator.set_view_bounds(NDC)
        if do_update:
            self.update_visuals()
        self._last_log_zoom = (1, 1)
        self._last_pan = (0, 0)

    def _create_visuals(self):
        if self.show_x:
            self.xvisual = LineVisual()
            self.txvisual = TextVisual()
            _fix_coordinate_in_visual(self.xvisual, 'y')
            _fix_coordinate_in_visual(self.txvisual, 'y')

        if self.show_y:
            self.yvisual = LineVisual()
            self.tyvisual = TextVisual()
            _fix_coordinate_in_visual(self.yvisual, 'x')
            _fix_coordinate_in_visual(self.tyvisual, 'x')

    def update_visuals(self):
        # Get the text data.
        xtext, ytext = self.locator.xtext, self.locator.ytext
        # GPU data for the grid.
        xdata, ydata = _set_line_data(self.locator.xticks_view, self.locator.yticks_view)
        # Position of the text in view coordinates.
        xpos, ypos = xdata[:, :2], ydata[:, 2:]

        # Set the visuals data.
        if self.show_x:
            self.xvisual.set_data(xdata, color=self.color)
            self.txvisual.set_data(pos=xpos, text=xtext, anchor=(0, +1.02))

        if self.show_y:
            self.yvisual.set_data(ydata, color=self.color)
            self.tyvisual.set_data(pos=ypos, text=ytext, anchor=(-1.02, 0))

    def attach(self, canvas):
        # Only attach once to avoid binding lots of events.
        if self._attached:
            return
        self._attached = canvas

        visuals = []

        if self.show_x:
            visuals += [self.xvisual, self.txvisual]
        if self.show_y:
            visuals += [self.yvisual, self.tyvisual]
        for visual in visuals:
            # Exclude the axes visual from the interact/layout, but keep the PanZoom.
            interact = getattr(canvas, 'interact', None)
            exclude_origins = (interact,) if interact else ()
            canvas.add_visual(
                visual, clearable=False, exclude_origins=exclude_origins)

        self.locator.set_view_bounds(NDC)
        self.update_visuals()

        @connect(sender=canvas)
        def on_resize(sender, w, h):
            nbinsx, nbinsy = get_nbins(w, h)
            self.locator.set_nbins(nbinsx, nbinsy)
            self.locator.set_view_bounds(canvas.panzoom.get_range())
            self.update_visuals()

        @connect(sender=canvas.panzoom)
        def on_zoom(sender, zoom):
            zx, zy = zoom
            ix, iy = _quant_zoom(zx), _quant_zoom(zy)
            if (ix, iy) != self._last_log_zoom:
                self._last_log_zoom = ix, iy
                self.locator.set_view_bounds(canvas.panzoom.get_range())
                self.update_visuals()

        @connect(sender=canvas.panzoom)
        def on_pan(sender, pan):
            px, py = pan
            zx, zy = canvas.panzoom.zoom
            tx, ty = int(px * zx), int(py * zy)
            if (tx, ty) != self._last_pan:
                self._last_pan = tx, ty
                self.locator.set_view_bounds(canvas.panzoom.get_range())
                self.update_visuals()
