# -*- coding: utf-8 -*-

"""Axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from matplotlib.ticker import MaxNLocator


from .transform import NDC, Range, _fix_coordinate_in_visual
from .visuals import LineVisual, TextVisual
from phylib import connect
from phylib.utils._types import _is_integer
from phy.gui.qt import is_high_dpi


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class AxisLocator(object):
    """Determine the location of ticks in a view.

    Constructor
    -----------

    nbinsx : int
        Number of ticks on the x axis.
    nbinsy : int
        Number of ticks on the y axis.
    data_bounds : 4-tuple
        Initial coordinates of the viewport, as (xmin, ymin, xmax, ymax), in data coordinates.
        These are the data coordinates of the lower left and upper right points of the window.

    """

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
        """Change the number of bins on the x and y axes."""
        nbinsx = self._bins_margin * nbinsx if _is_integer(nbinsx) else self._default_nbinsx
        nbinsy = self._bins_margin * nbinsy if _is_integer(nbinsy) else self._default_nbinsy
        self.locx = MaxNLocator(nbins=nbinsx, steps=self._default_steps)
        self.locy = MaxNLocator(nbins=nbinsy, steps=self._default_steps)

    def _transform_ticks(self, xticks, yticks):
        """Transform ticks from data coordinates to view coordinates."""
        nx, ny = len(xticks), len(yticks)
        arr = np.zeros((nx + ny, 2))
        arr[:nx, 0] = xticks
        arr[nx:, 1] = yticks
        out = self._tri.apply(arr)
        return out[:nx, 0], out[nx:, 1]

    def set_view_bounds(self, view_bounds=None):
        """Set the view bounds in normalized device coordinates. Used when panning and zooming.

        This method updates the following attributes:

        * xticks : the position of the ticks on the x axis
        * yticks : the position of the ticks on the y axis
        * xtext : the text of the ticks on the x axis
        * ytext : the text of the ticks on the y axis

        """
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
        fmt = '%.9g'
        self.xtext = [fmt % v for v in self.xticks]
        self.ytext = [fmt % v for v in self.yticks]


#------------------------------------------------------------------------------
# Axes visual
#------------------------------------------------------------------------------


def _set_line_data(xticks, yticks):
    """Return the positions of the line ticks."""
    xdata = np.c_[xticks, -np.ones(len(xticks)), xticks, np.ones(len(xticks))]
    ydata = np.c_[-np.ones(len(yticks)), yticks, np.ones(len(yticks)), yticks]
    return xdata, ydata


def get_nbins(w, h):
    """Return a sensible number of bins on the x and y axes as a function of the window size."""
    if is_high_dpi():  # pragma: no cover
        w, h = w // 2, h // 2
    return max(1, w // 150), max(1, h // 80)


def _quant_zoom(z):
    """Return the zoom level as a positive or negative integer."""
    if z == 0:
        return 0  # pragma: no cover
    return int(z) if z >= 1 else -int(1. / z)


class Axes(object):
    """Dynamic axes that move along the camera when panning and zooming.

    Constructor
    -----------

    data_bounds : 4-tuple
        The data coordinates of the initial viewport (when there is no panning and zooming).
    color : 4-tuple
        Color of the grid.
    show_x : boolean
        Whether to show the vertical grid lines.
    show_y : boolean
        Whether to show the horizontal grid lines.

    """
    default_color = (1, 1, 1, .25)

    def __init__(self, data_bounds=None, color=None, show_x=True, show_y=True):
        self.show_x = show_x
        self.show_y = show_y
        self.reset_data_bounds(data_bounds, do_update=False)
        self._create_visuals()
        self.color = color or self.default_color
        self._attached = None

    def reset_data_bounds(self, data_bounds, do_update=True):
        """Reset the bounds of the view in data coordinates.

        Used when the view is recreated from scratch.

        """
        self.locator = AxisLocator(data_bounds=data_bounds)
        self.locator.set_view_bounds(NDC)
        if do_update:
            self.update_visuals()
        self._last_log_zoom = (1, 1)
        self._last_pan = (0, 0)

    def _create_visuals(self):
        """Create the line and text visuals on the x and/or y axes."""
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
        """Update the grid and text visuals after updating the axis locator."""

        # Get the text data.
        xtext, ytext = self.locator.xtext, self.locator.ytext
        # GPU data for the grid.
        xdata, ydata = _set_line_data(self.locator.xticks_view, self.locator.yticks_view)
        # Position of the text in view coordinates.
        xpos, ypos = xdata[:, :2], ydata[:, 2:]

        # Set the visuals data.
        if self.show_x:
            self.xvisual.set_data(xdata, color=self.color)
            self.txvisual.set_data(pos=xpos, text=xtext, anchor=(0, +1))

        if self.show_y:
            self.yvisual.set_data(ydata, color=self.color)
            self.tyvisual.set_data(pos=ypos, text=ytext, anchor=(-1, 0))

    def attach(self, canvas):
        """Add the axes to a canvas.

        Add the grid and text visuals to the canvas, and attach to the pan and zoom events
        raised by the canvas.

        """

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
            return self._update_zoom(zoom)

        @connect(sender=canvas.panzoom)
        def on_pan(sender, pan):
            return self._update_pan(pan)

    def _update_zoom(self, zoom, force=False):
        zx, zy = zoom
        ix, iy = _quant_zoom(zx), _quant_zoom(zy)
        if force or (ix, iy) != self._last_log_zoom:
            self._last_log_zoom = ix, iy
            self.locator.set_view_bounds(self._attached.panzoom.get_range())
            self.update_visuals()

    def _update_pan(self, pan, force=False):
        px, py = pan
        zx, zy = self._attached.panzoom.zoom
        tx, ty = int(px * zx), int(py * zy)
        if force or (tx, ty) != self._last_pan:
            self._last_pan = tx, ty
            self.locator.set_view_bounds(self._attached.panzoom.get_range())
            self.update_visuals()
