# -*- coding: utf-8 -*-

"""Plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict
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
from .utils import BatchAccumulator
from phylib.utils._types import _as_tuple

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Plotting interface
#------------------------------------------------------------------------------

class PlotCanvas(BaseCanvas):
    """Plotting canvas."""
    _default_box_index = (0,)
    interact = None
    n_plots = 1
    has_panzoom = True
    has_axes = False
    has_lasso = False
    constrain_bounds = (-2, -2, +2, +2)
    _enabled = False

    def __init__(self, *args, **kwargs):
        super(PlotCanvas, self).__init__(*args, **kwargs)
        self._acc = defaultdict(BatchAccumulator)  # dict visual_cls => BatchAccumulator()

    def _enable(self):
        self._enabled = True
        if self.has_panzoom:
            self.enable_panzoom()
        if self.has_axes:
            self.enable_axes()
        if self.has_lasso:
            self.enable_lasso()

    def set_layout(
            self, layout=None, shape=None, n_plots=None, origin=None,
            box_bounds=None, box_pos=None, box_size=None, has_clip=None):

        self.layout = layout

        # Constrain pan zoom.
        if layout == 'grid':
            self._default_box_index = (0, 0)
            self.grid = Grid(shape, has_clip=has_clip)
            self.grid.attach(self)
            self.interact = self.grid
            self.constrain_bounds = (-1, -1, +1, +1)

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
            self.stacked = Stacked(n_plots, origin=origin)
            self.stacked.attach(self)
            self.interact = self.stacked
            self.constrain_bounds = (-1, -1, +1, +1)

        if layout == 'grid' and shape is not None:
            self.interact.add_boxes(self)

    def __getitem__(self, box_index):
        self._default_box_index = _as_tuple(box_index)
        return self

    @property
    def canvas(self):
        return self

    def add_visual(self, *args, **kwargs):
        if not self._enabled:
            self._enable()
        return super(PlotCanvas, self).add_visual(*args, **kwargs)

    def add_one(self, visual, *args, box_index=None, **kwargs):
        # Finalize batch.
        cls = visual.__class__
        # WARNING: self._acc[cls] should not be silently created here
        # when accessing self._acc[cls] of a defaultdict.
        # If the first part of the if test fails, then the second part
        # is not even run.
        if cls in self._acc and self._acc[cls].items:
            kwargs.update(self._acc[cls].data)
            box_index = box_index if box_index is not None else self._acc[cls].box_index
            self._acc[cls].reset()
        else:
            box_index = box_index if box_index is not None else self._default_box_index
        self.add_visual(
            visual,
            clearable=kwargs.pop('clearable', True),
            key=kwargs.pop('key', None),
        )
        # Remove box_index from the kwargs updated with the BatchAccumulator.
        kwargs.pop('box_index', None)
        data = visual.set_data(*args, **kwargs)
        if self.interact:
            visual.set_box_index(box_index, data=data)
        return visual

    def add_batch(self, visual_cls, box_index=None, **kwargs):
        # box_index scalar or vector
        b = visual_cls.validate(**kwargs)
        if box_index is not None:  # pragma: no cover
            b.box_index = box_index
        else:
            n = visual_cls.vertex_count(**kwargs)
            b.box_index = np.tile(np.atleast_2d(self._default_box_index), (n, 1))
        return self._acc[visual_cls].add(b)

    # Plot methods
    #--------------------------------------------------------------------------

    def scatter(self, *args, **kwargs):
        return self.add_one(ScatterVisual(marker=kwargs.pop('marker', None)), *args, **kwargs)

    def uscatter(self, *args, **kwargs):
        return self.add_one(UniformScatterVisual(
            marker=kwargs.pop('marker', None),
            color=kwargs.pop('color', None),
            size=kwargs.pop('size', None)), *args, **kwargs)

    def plot(self, *args, **kwargs):
        return self.add_one(PlotVisual(), *args, **kwargs)

    def uplot(self, *args, **kwargs):
        return self.add_one(UniformPlotVisual(color=kwargs.pop('color', None)), *args, **kwargs)

    def lines(self, *args, **kwargs):
        return self.add_one(LineVisual(), *args, **kwargs)

    def text(self, *args, **kwargs):
        return self.add_one(TextVisual(color=kwargs.pop('color', None)), *args, **kwargs)

    def polygon(self, *args, **kwargs):
        return self.add_one(PolygonVisual(), *args, **kwargs)

    def hist(self, *args, **kwargs):
        return self.add_one(HistogramVisual(), *args, **kwargs)

    # Batch methods
    #--------------------------------------------------------------------------

    def text_batch(self, **kwargs):
        # box_index scalar or vector
        b = TextVisual.validate(**kwargs)
        b.box_index = kwargs.pop('box_index', self._default_box_index)
        if isinstance(b.box_index, tuple):
            b.box_index = [b.box_index]
        return self._acc[TextVisual].add(b, noconcat=('text', 'box_index'))

    def uscatter_batch(self, **kwargs):
        return self.add_batch(UniformScatterVisual, **kwargs)

    def lines_batch(self, **kwargs):
        return self.add_batch(LineVisual, **kwargs)

    def hist_batch(self, **kwargs):
        return self.add_batch(HistogramVisual, **kwargs)

    def plot_batch(self, **kwargs):
        # box_index scalar or vector
        box_index = kwargs.pop('box_index', None)
        b = PlotVisual.validate(**kwargs)
        if box_index is not None:  # pragma: no cover
            b.box_index = box_index
        else:
            n = PlotVisual.vertex_count(**kwargs)
            b.box_index = np.tile(np.atleast_2d(self._default_box_index), (n, 1))
        return self._acc[PlotVisual].add(b, noconcat=('x', 'y'))

    # Enable methods
    #--------------------------------------------------------------------------

    def enable_panzoom(self):
        self.panzoom = PanZoom(aspect=None, constrain_bounds=self.constrain_bounds)
        self.panzoom.attach(self)

    def enable_lasso(self):
        self.lasso = Lasso()
        self.lasso.attach(self)

    def enable_axes(self, data_bounds=None, show_x=True, show_y=True):
        self.axes = Axes(data_bounds=data_bounds, show_x=show_x, show_y=show_y)
        self.axes.attach(self)


#------------------------------------------------------------------------------
# Matplotlib plotting interface
#------------------------------------------------------------------------------

def zoom_fun(ax, event):  # pragma: no cover
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
    _default_box_index = (0,)
    gui = None
    _shown = False
    axes = None

    def __init__(self, *args, **kwargs):
        plt.style.use('dark_background')
        mpl.rcParams['toolbar'] = 'None'
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[DEFAULT_COLOR])
        self.figure = plt.figure()
        self.subplots()

    def set_layout(
            self, layout=None, shape=None, n_plots=None, origin=None,
            box_bounds=None, box_pos=None, box_size=None):

        self.layout = layout

        # Constrain pan zoom.
        if layout == 'grid':
            self.subplots(shape[0], shape[1])
            self._default_box_index = (0, 0)

        elif layout == 'boxed':  # pragma: no cover
            self.n_plots = (len(box_bounds)
                            if box_bounds is not None else len(box_pos))
            # self.boxed = Boxed(box_bounds=box_bounds,
            #                    box_pos=box_pos,
            #                    box_size=box_size)
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
            zoom_fun(ax, event)
            self.show()

        self.canvas.mpl_connect('scroll_event', on_zoom)

    def __getitem__(self, box_index):
        self._default_box_index = _as_tuple(box_index)
        return self

    @property
    def ax(self):
        if len(self._default_box_index) == 1:
            return self.axes[0, self._default_box_index[0]]
        else:
            return self.axes[self._default_box_index]

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
            self.figure.show()
        self._shown = True

    def close(self):
        self.canvas.close()
        plt.close(self.figure)
