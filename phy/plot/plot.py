# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .axes import Axes
from .base import BaseCanvas
from .interact import Grid, Boxed, Stacked, Lasso
from .panzoom import PanZoom
from .visuals import (
    ScatterVisual, UniformScatterVisual, PlotVisual, UniformPlotVisual,
    HistogramVisual, TextVisual, LineVisual, PolygonVisual,
    DEFAULT_COLOR)
from .transform import NDC
from phylib.utils._types import _as_tuple

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class PlotCanvas(BaseCanvas):
    """Plotting canvas that supports different layouts, subplots, lasso, axes, panzoom."""

    _current_box_index = (0,)
    interact = None
    n_plots = 1
    has_panzoom = True
    has_axes = False
    has_lasso = False
    constrain_bounds = (-2, -2, +2, +2)
    _enabled = False

    def __init__(self, *args, **kwargs):
        super(PlotCanvas, self).__init__(*args, **kwargs)

    def _enable(self):
        """Enable panzoom, axes, and lasso if required."""
        self._enabled = True
        if self.has_panzoom:
            self.enable_panzoom()
        if self.has_axes:
            self.enable_axes()
        if self.has_lasso:
            self.enable_lasso()

    def set_layout(
            self, layout=None, shape=None, n_plots=None, origin=None,
            box_pos=None, has_clip=True):
        """Set the plot layout: grid, boxed, stacked, or None."""

        self.layout = layout

        # Constrain pan zoom.
        if layout == 'grid':
            self._current_box_index = (0, 0)
            self.grid = Grid(shape, has_clip=has_clip)
            self.grid.attach(self)
            self.interact = self.grid

        elif layout == 'boxed':
            self.n_plots = len(box_pos)
            self.boxed = Boxed(box_pos=box_pos)
            self.boxed.attach(self)
            self.interact = self.boxed

        elif layout == 'stacked':
            self.n_plots = n_plots
            self.stacked = Stacked(n_plots, origin=origin)
            self.stacked.attach(self)
            self.interact = self.stacked

        if layout == 'grid' and shape is not None:
            self.interact.add_boxes(self)

    def __getitem__(self, box_index):
        self._current_box_index = _as_tuple(box_index)
        return self

    @property
    def canvas(self):
        return self

    def add_visual(self, visual, *args, **kwargs):
        """Add a visual and possibly set some data directly.

        Parameters
        ----------

        visual : Visual
        clearable : True
            Whether the visual should be deleted when calling `canvas.clear()`.
        exclude_origins : list-like
            List of interact instances that should not apply to that visual. For example, use to
            add a visual outside of the subplots, or with no support for pan and zoom.
        key : str
            An optional key to identify a visual

        """
        if not self._enabled:
            self._enable()
        # The visual is not added again if it has already been added, in which case
        # the following call is a no-op.
        super(PlotCanvas, self).add_visual(
            visual,
            # Remove special reserved keywords from kwargs, which is otherwise supposed to
            # contain data for visual.set_data().
            clearable=kwargs.pop('clearable', True),
            key=kwargs.pop('key', None),
            exclude_origins=kwargs.pop('exclude_origins', ()),
        )
        self.update_visual(visual, *args, **kwargs)

    def update_visual(self, visual, *args, **kwargs):
        """Set the data of a visual, standalone or at the end of a batch."""
        if not self._enabled:  # pragma: no cover
            self._enable()
        # If a batch session has been initiated in the visual, add the data from the
        # visual's BatchAccumulator.
        if visual._acc.items:
            kwargs.update(visual._acc.data)
            # If the batch accumulator has box_index, we get it in kwargs now.
        # We remove the box_index before calling set_data().
        box_index = kwargs.pop('box_index', None)
        # If no data was obtained at this point, we return.
        if box_index is None and not kwargs:
            return visual
        # If kwargs is not empty, we set the data on the visual.
        data = visual.set_data(*args, **kwargs) if kwargs else None
        # Finally, we may need to set the box index.
        # box_index could be specified directly to add_visual, or it could have been
        # constructed in the batch, or finally it should just be the current box index
        # by default.
        if self.interact and data:
            box_index = box_index if box_index is not None else self._current_box_index
            visual.set_box_index(box_index, data=data)
        return visual

    # Plot methods
    #--------------------------------------------------------------------------

    def scatter(self, *args, **kwargs):
        """Add a standalone (no batch) scatter plot."""
        return self.add_visual(ScatterVisual(marker=kwargs.pop('marker', None)), *args, **kwargs)

    def uscatter(self, *args, **kwargs):
        """Add a standalone (no batch) uniform scatter plot."""
        return self.add_visual(UniformScatterVisual(
            marker=kwargs.pop('marker', None),
            color=kwargs.pop('color', None),
            size=kwargs.pop('size', None)), *args, **kwargs)

    def plot(self, *args, **kwargs):
        """Add a standalone (no batch) plot."""
        return self.add_visual(PlotVisual(), *args, **kwargs)

    def uplot(self, *args, **kwargs):
        """Add a standalone (no batch) uniform plot."""
        return self.add_visual(UniformPlotVisual(color=kwargs.pop('color', None)), *args, **kwargs)

    def lines(self, *args, **kwargs):
        """Add a standalone (no batch) line plot."""
        return self.add_visual(LineVisual(), *args, **kwargs)

    def text(self, *args, **kwargs):
        """Add a standalone (no batch) text plot."""
        return self.add_visual(TextVisual(color=kwargs.pop('color', None)), *args, **kwargs)

    def polygon(self, *args, **kwargs):
        """Add a standalone (no batch) polygon plot."""
        return self.add_visual(PolygonVisual(), *args, **kwargs)

    def hist(self, *args, **kwargs):
        """Add a standalone (no batch) histogram plot."""
        return self.add_visual(HistogramVisual(), *args, **kwargs)

    # Enable methods
    #--------------------------------------------------------------------------

    def enable_panzoom(self):
        """Enable pan zoom in the canvas."""
        self.panzoom = PanZoom(aspect=None, constrain_bounds=self.constrain_bounds)
        self.panzoom.attach(self)

    def enable_lasso(self):
        """Enable lasso in the canvas."""
        self.lasso = Lasso()
        self.lasso.attach(self)

    def enable_axes(self, data_bounds=None, show_x=True, show_y=True):
        """Show axes in the canvas."""
        self.axes = Axes(data_bounds=data_bounds, show_x=show_x, show_y=show_y)
        self.axes.attach(self)


#------------------------------------------------------------------------------
# Matplotlib plotting interface
#------------------------------------------------------------------------------

def _zoom_fun(ax, event):  # pragma: no cover
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata
    ydata = event.ydata
    if xdata is None or ydata is None:
        return
    x_left = xdata - cur_xlim[0]
    x_right = cur_xlim[1] - xdata
    y_top = ydata - cur_ylim[0]
    y_bottom = cur_ylim[1] - ydata
    k = 1.3
    scale_factor = {'up': 1. / k, 'down': k}.get(event.button, 1.)
    ax.set_xlim([xdata - x_left * scale_factor,
                 xdata + x_right * scale_factor])
    ax.set_ylim([ydata - y_top * scale_factor,
                 ydata + y_bottom * scale_factor])


_MPL_MARKER = {
    'arrow': '>',
    'asterisk': '*',
    'chevron': '^',
    'club': 'd',
    'cross': 'x',
    'diamond': 'D',
    'disc': 'o',
    'ellipse': 'o',
    'hbar': '_',
    'square': 's',
    'triangle': '^',
    'vbar': '|',
}


class PlotCanvasMpl(object):
    """Matplotlib backend for a plot canvas (incomplete, work in progress)."""

    _current_box_index = (0,)
    gui = None
    _shown = False
    axes = None

    def __init__(self, *args, **kwargs):
        plt.style.use('dark_background')
        mpl.rcParams['toolbar'] = 'None'
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[DEFAULT_COLOR])
        self.figure = plt.figure()
        self.subplots()

    def set_layout(self, layout=None, shape=None, n_plots=None, origin=None, box_pos=None):

        self.layout = layout

        # Constrain pan zoom.
        if layout == 'grid':
            self.subplots(shape[0], shape[1])
            self._current_box_index = (0, 0)

        elif layout == 'boxed':  # pragma: no cover
            self.n_plots = len(box_pos)
            # self.boxed = Boxed(box_pos=box_pos)
            # TODO
            raise NotImplementedError()

        elif layout == 'stacked':  # pragma: no cover
            self.n_plots = n_plots
            # self.stacked = Stacked(n_plots, margin=.1, origin=origin)
            # TODO
            raise NotImplementedError()

    def subplots(self, nrows=1, ncols=1, **kwargs):
        self.figure.clf()
        self.axes = self.figure.subplots(nrows, ncols, squeeze=False, **kwargs)
        for ax in self.iter_ax():
            self.config_ax(ax)
        return self.axes

    def iter_ax(self):
        for ax in self.axes.flat:
            yield ax

    def config_ax(self, ax):
        xaxis = ax.get_xaxis()
        yaxis = ax.get_yaxis()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        xaxis.set_ticks_position('bottom')
        xaxis.set_tick_params(direction='out')

        yaxis.set_ticks_position('left')
        yaxis.set_tick_params(direction='out')

        ax.grid(color='w', alpha=.2)

        def on_zoom(event):  # pragma: no cover
            _zoom_fun(ax, event)
            self.show()

        self.canvas.mpl_connect('scroll_event', on_zoom)

    def __getitem__(self, box_index):
        self._current_box_index = _as_tuple(box_index)
        return self

    def attach_events(self, view):
        pass

    def set_lazy(self, lazy):
        pass

    @property
    def ax(self):
        if len(self._current_box_index) == 1:
            return self.axes[0, self._current_box_index[0]]
        else:
            return self.axes[self._current_box_index]

    def enable_axes(self):
        pass

    def enable_lasso(self):
        pass

    def enable_panzoom(self):
        pass

    def set_data_bounds(self, data_bounds):
        data_bounds = data_bounds or NDC
        assert len(data_bounds) == 4
        x0, y0, x1, y1 = data_bounds
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def scatter(
            self, x=None, y=None, pos=None, color=None,
            size=None, depth=None, data_bounds=None, marker=None):
        self.ax.scatter(x, y, c=color, s=size, marker=_MPL_MARKER.get(marker, 'o'))
        self.set_data_bounds(data_bounds)

    def plot(self, x=None, y=None, color=None, depth=None, data_bounds=None):
        self.ax.plot(x, y, c=color)
        self.set_data_bounds(data_bounds)

    def hist(self, hist=None, color=None, ylim=None):
        assert hist is not None
        n = len(hist)
        x = np.linspace(-1, 1, n)
        self.ax.bar(x, hist, width=2. / (n - 1), color=color)
        self.set_data_bounds((-1, 0, +1, ylim))

    def lines(self, pos=None, color=None, data_bounds=None):
        pos = np.atleast_2d(pos)
        x0, y0, x1, y1 = pos.T
        x = np.r_[x0, x1]
        y = np.r_[y0, y1]
        self.ax.plot(x, y, c=color)
        self.set_data_bounds(data_bounds)

    def text(self, pos=None, text=None, anchor=None,
             data_bounds=None, color=None):
        pos = np.atleast_2d(pos)
        self.ax.text(pos[:, 0], pos[:, 1], text, color=color or 'w')
        self.set_data_bounds(data_bounds)

    def polygon(self, pos=None, data_bounds=None):
        self.ax.plot(pos[:, 0], pos[:, 1])
        self.set_data_bounds(data_bounds)

    @property
    def canvas(self):
        return self.figure.canvas

    def attach(self, gui):
        self.gui = gui
        self.nav = NavigationToolbar(self.canvas, gui, coordinates=False)
        self.nav.pan()

    def clear(self):
        for ax in self.iter_ax():
            ax.clear()

    def show(self):
        self.canvas.draw()
        if not self.gui and not self._shown:
            self.nav = NavigationToolbar(self.canvas, None, coordinates=False)
            self.nav.pan()
        self._shown = True

    def update(self):  # pragma: no cover
        return self.show()

    def close(self):
        self.canvas.close()
        plt.close(self.figure)
