# -*- coding: utf-8 -*-

"""Test visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..visuals import (ScatterVisual, PlotVisual, HistogramVisual,
                       LineVisual, PolygonVisual, TextVisual,
                       UniformPlotVisual, UniformScatterVisual,
                       )
from phy.utils._color import _random_color


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _test_visual(qtbot, c, v, stop=False, **kwargs):
    c.add_visual(v)
    data = v.validate(**kwargs)
    assert v.vertex_count(**data) >= 0
    v.set_data(**kwargs)
    c.show()
    qtbot.waitForWindowShown(c.native)
    if stop:  # pragma: no cover
        qtbot.stop()
    c.close()


#------------------------------------------------------------------------------
# Test scatter visual
#------------------------------------------------------------------------------

def test_scatter_empty(qtbot, canvas):
    _test_visual(qtbot, canvas, ScatterVisual(), x=np.zeros(0), y=np.zeros(0))


def test_scatter_markers(qtbot, canvas_pz):

    n = 100
    x = .2 * np.random.randn(n)
    y = .2 * np.random.randn(n)

    _test_visual(qtbot, canvas_pz,
                 ScatterVisual(marker='vbar'),
                 x=x, y=y)


def test_scatter_custom(qtbot, canvas_pz):

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    # Random colors.
    c = np.random.uniform(.4, .7, size=(n, 4))
    c[:, -1] = .5

    # Random sizes
    s = 5 + 20 * np.random.rand(n)

    _test_visual(qtbot, canvas_pz, ScatterVisual(),
                 pos=pos, color=c, size=s)


#------------------------------------------------------------------------------
# Test uniform scatter visual
#------------------------------------------------------------------------------

def test_uniform_scatter_empty(qtbot, canvas):
    _test_visual(qtbot, canvas, UniformScatterVisual(),
                 x=np.zeros(0), y=np.zeros(0))


def test_uniform_scatter_markers(qtbot, canvas_pz):

    n = 100
    x = .2 * np.random.randn(n)
    y = .2 * np.random.randn(n)

    _test_visual(qtbot, canvas_pz,
                 UniformScatterVisual(marker='vbar'),
                 x=x, y=y)


def test_uniform_scatter_custom(qtbot, canvas_pz):

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    _test_visual(qtbot, canvas_pz,
                 UniformScatterVisual(color=_random_color() + (.5,),
                                      size=10.,
                                      ),
                 pos=pos,
                 masks=np.linspace(0., 1., n),
                 data_bounds=None,
                 )


#------------------------------------------------------------------------------
# Test plot visual
#------------------------------------------------------------------------------

def test_plot_empty(qtbot, canvas):
    y = np.zeros((1, 0))
    _test_visual(qtbot, canvas, PlotVisual(),
                 y=y)


def test_plot_0(qtbot, canvas_pz):
    y = np.zeros((1, 10))
    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 y=y)


def test_plot_1(qtbot, canvas_pz):
    y = .2 * np.random.randn(10)
    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 y=y)


def test_plot_2(qtbot, canvas_pz):

    n_signals = 50
    n_samples = 10
    y = 20 * np.random.randn(n_signals, n_samples)

    # Signal colors.
    c = np.random.uniform(.5, 1, size=(n_signals, 4))
    c[:, 3] = .5

    # Depth.
    depth = np.linspace(0., -1., n_signals)

    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 y=y, depth=depth,
                 data_bounds=[-1, -50, 1, 50],
                 color=c,
                 )


def test_plot_list(qtbot, canvas_pz):
    y = [np.random.randn(i) for i in (5, 20)]
    c = np.random.uniform(.5, 1, size=(2, 4))
    c[:, 3] = .5

    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 y=y, color=c)


#------------------------------------------------------------------------------
# Test uniform plot visual
#------------------------------------------------------------------------------

def test_uniform_plot_empty(qtbot, canvas):
    y = np.zeros((1, 0))
    _test_visual(qtbot, canvas, UniformPlotVisual(),
                 y=y)


def test_uniform_plot_0(qtbot, canvas_pz):
    y = np.zeros((1, 10))
    _test_visual(qtbot, canvas_pz, UniformPlotVisual(),
                 y=y)


def test_uniform_plot_1(qtbot, canvas_pz):
    y = .2 * np.random.randn(10)
    _test_visual(qtbot, canvas_pz,
                 UniformPlotVisual(),
                 y=y,
                 masks=.5,
                 data_bounds=None,
                 )


def test_uniform_plot_list(qtbot, canvas_pz):
    y = [np.random.randn(i) for i in (5, 20)]

    _test_visual(qtbot, canvas_pz,
                 UniformPlotVisual(color=(1., 0., 0., 1.)),
                 y=y,
                 masks=[.1, .9],
                 )


#------------------------------------------------------------------------------
# Test histogram visual
#------------------------------------------------------------------------------

def test_histogram_empty(qtbot, canvas):
    hist = np.zeros((1, 0))
    _test_visual(qtbot, canvas, HistogramVisual(),
                 hist=hist)


def test_histogram_0(qtbot, canvas_pz):
    hist = np.zeros((10,))
    _test_visual(qtbot, canvas_pz, HistogramVisual(),
                 hist=hist)


def test_histogram_1(qtbot, canvas_pz):
    hist = np.random.rand(1, 10)
    _test_visual(qtbot, canvas_pz, HistogramVisual(),
                 hist=hist)


def test_histogram_2(qtbot, canvas_pz):

    n_hists = 5
    hist = np.random.rand(n_hists, 21)

    # Histogram colors.
    c = np.random.uniform(.3, .6, size=(n_hists, 4))
    c[:, 3] = 1

    _test_visual(qtbot, canvas_pz, HistogramVisual(),
                 hist=hist, color=c, ylim=2 * np.ones(n_hists))


#------------------------------------------------------------------------------
# Test line visual
#------------------------------------------------------------------------------

def test_line_empty(qtbot, canvas):
    pos = np.zeros((0, 4))
    _test_visual(qtbot, canvas, LineVisual(), pos=pos)


def test_line_0(qtbot, canvas_pz):
    n = 10
    y = np.linspace(-.5, .5, 10)
    pos = np.c_[-np.ones(n), y, np.ones(n), y]
    color = np.random.uniform(.5, .9, (n, 4))
    _test_visual(qtbot, canvas_pz, LineVisual(),
                 pos=pos, color=color, data_bounds=[-1, -1, 1, 1])


#------------------------------------------------------------------------------
# Test polygon visual
#------------------------------------------------------------------------------

def test_polygon_empty(qtbot, canvas):
    pos = np.zeros((0, 2))
    _test_visual(qtbot, canvas, PolygonVisual(), pos=pos)


def test_polygon_0(qtbot, canvas_pz):
    n = 9
    x = .5 * np.cos(np.linspace(0., 2 * np.pi, n))
    y = .5 * np.sin(np.linspace(0., 2 * np.pi, n))
    pos = np.c_[x, y]
    _test_visual(qtbot, canvas_pz, PolygonVisual(), pos=pos)


#------------------------------------------------------------------------------
# Test text visual
#------------------------------------------------------------------------------

def test_text_empty(qtbot, canvas):
    pos = np.zeros((0, 2))
    _test_visual(qtbot, canvas, TextVisual(), pos=pos, text=[])
    _test_visual(qtbot, canvas, TextVisual())


def test_text_0(qtbot, canvas_pz):
    text = '0123456789'
    text = [text[:n] for n in range(1, 11)]

    pos = np.c_[np.linspace(-.5, .5, 10), np.linspace(-.5, .5, 10)]

    _test_visual(qtbot, canvas_pz, TextVisual(),
                 pos=pos, text=text)


def test_text_1(qtbot, canvas_pz):
    c = canvas_pz

    text = ['--x--'] * 5
    pos = [[0, 0], [-.5, +.5], [+.5, +.5], [-.5, -.5], [+.5, -.5]]
    anchor = [[0, 0], [-1, +1], [+1, +1], [-1, -1], [+1, -1]]

    v = TextVisual()
    c.add_visual(v)
    v.set_data(pos=pos, text=text, anchor=anchor, data_bounds=None)

    v = ScatterVisual()
    c.add_visual(v)
    v.set_data(pos=pos, data_bounds=None)

    c.show()
    qtbot.waitForWindowShown(c.native)

    # qtbot.stop()
    c.close()
