# -*- coding: utf-8 -*-

"""Test visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np

from ..visuals import (
    ScatterVisual, PatchVisual, PlotVisual, HistogramVisual, LineVisual,
    LineAggGeomVisual, PlotAggVisual,
    PolygonVisual, TextVisual, ImageVisual, UniformPlotVisual, UniformScatterVisual)
from ..transform import NDC, Rotate, range_transform
from phy.utils.color import _random_color


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _test_visual(qtbot, c, v, stop=False, **kwargs):
    c.add_visual(v)
    data = v.validate(**kwargs)
    assert v.vertex_count(**data) >= 0
    v.set_data(**kwargs)
    c.show()
    qtbot.waitForWindowShown(c)
    if os.environ.get('PHY_TEST_STOP', None) or stop:  # pragma: no cover
        qtbot.stop()
    v.close()
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

    _test_visual(qtbot, canvas_pz, ScatterVisual(marker='vbar'), x=x, y=y, data_bounds='auto')


def test_scatter_custom(qtbot, canvas_pz):

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    # Random colors.
    c = np.random.uniform(.4, .7, size=(n, 4))
    c[:, -1] = .5

    # Random sizes
    s = 5 + 20 * np.random.rand(n)

    _test_visual(qtbot, canvas_pz, ScatterVisual(), pos=pos, color=c, size=s)


#------------------------------------------------------------------------------
# Test patch visual
#------------------------------------------------------------------------------

def test_patch_empty(qtbot, canvas):
    _test_visual(qtbot, canvas, PatchVisual(), x=np.zeros(0), y=np.zeros(0))


def test_patch_1(qtbot, canvas_pz):

    n = 100
    x = .2 * np.random.randn(n)
    y = .2 * np.random.randn(n)

    _test_visual(qtbot, canvas_pz, PatchVisual(), x=x, y=y, data_bounds='auto')


def test_patch_2(qtbot, canvas_pz):

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    # Random colors.
    c = np.random.uniform(.4, .7, size=(n, 4))
    c[:, -1] = .5

    v = PatchVisual(primitive_type='triangles')
    canvas_pz.add_visual(v)
    v.set_data(pos=pos, color=c)
    canvas_pz.show()
    qtbot.waitForWindowShown(canvas_pz)
    v.set_color((1, 1, 0, 1))
    canvas_pz.update()
    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    v.close()
    canvas_pz.close()


#------------------------------------------------------------------------------
# Test uniform scatter visual
#------------------------------------------------------------------------------

def test_uniform_scatter_empty(qtbot, canvas):
    _test_visual(qtbot, canvas, UniformScatterVisual(), x=np.zeros(0), y=np.zeros(0))


def test_uniform_scatter_markers(qtbot, canvas_pz):

    n = 100
    x = .2 * np.random.randn(n)
    y = .2 * np.random.randn(n)

    _test_visual(
        qtbot, canvas_pz, UniformScatterVisual(marker='vbar'), x=x, y=y, data_bounds='auto')


def test_uniform_scatter_custom(qtbot, canvas_pz):

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    _test_visual(
        qtbot, canvas_pz, UniformScatterVisual(color=_random_color() + (.5,), size=10., ),
        pos=pos, masks=np.linspace(0., 1., n), data_bounds=None)


#------------------------------------------------------------------------------
# Test plot visual
#------------------------------------------------------------------------------

def test_plot_empty(qtbot, canvas):
    y = np.zeros((1, 0))
    _test_visual(qtbot, canvas, PlotVisual(),
                 y=y)


def test_plot_0(qtbot, canvas_pz):
    y = np.zeros((1, 10))
    _test_visual(qtbot, canvas_pz, PlotVisual(), y=y)


def test_plot_1(qtbot, canvas_pz):
    y = .2 * np.random.randn(10)
    _test_visual(qtbot, canvas_pz, PlotVisual(), y=y, data_bounds='auto')


def test_plot_color(qtbot, canvas_pz):
    v = PlotVisual()
    canvas_pz.add_visual(v)
    data = v.validate(y=.2 * np.random.randn(10), data_bounds='auto')
    assert v.vertex_count(**data) >= 0
    v.set_data(**data)
    v.set_color(np.random.uniform(low=.5, high=.9, size=(10, 4)))
    canvas_pz.show()
    qtbot.waitForWindowShown(canvas_pz)
    canvas_pz.close()


def test_plot_2(qtbot, canvas_pz):

    n_signals = 50
    n_samples = 10
    y = 20 * np.random.randn(n_signals, n_samples)

    # Signal colors.
    c = np.random.uniform(.5, 1, size=(n_signals, 4))
    c[:, 3] = .5

    # Depth.
    depth = np.linspace(0., -1., n_signals)

    _test_visual(
        qtbot, canvas_pz, PlotVisual(), y=y, depth=depth, data_bounds=[-1, -50, 1, 50], color=c)


def test_plot_list(qtbot, canvas_pz):
    y = [.25 * np.random.randn(i) for i in (5, 20, 50)]
    c = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    masks = [0., 0.5, 1.0]

    _test_visual(qtbot, canvas_pz, PlotVisual(), y=y, color=c, masks=masks)


#------------------------------------------------------------------------------
# Test uniform plot visual
#------------------------------------------------------------------------------

def test_uniform_plot_empty(qtbot, canvas):
    y = np.zeros((1, 0))
    _test_visual(qtbot, canvas, UniformPlotVisual(), y=y)


def test_uniform_plot_0(qtbot, canvas_pz):
    y = np.zeros((1, 10))
    _test_visual(qtbot, canvas_pz, UniformPlotVisual(), y=y)


def test_uniform_plot_1(qtbot, canvas_pz):
    y = .2 * np.random.randn(10)
    _test_visual(qtbot, canvas_pz, UniformPlotVisual(), y=y, masks=.5, data_bounds=NDC)


def test_uniform_plot_2(qtbot, canvas_pz):
    y = .2 * np.random.randn(10)
    _test_visual(qtbot, canvas_pz, UniformPlotVisual(), y=y, masks=.5, data_bounds='auto')


def test_uniform_plot_list(qtbot, canvas_pz):
    y = [np.random.randn(i) for i in (5, 20)]

    _test_visual(qtbot, canvas_pz, UniformPlotVisual(color=(1., 0., 0., 1.)), y=y, masks=[.1, .9])


#------------------------------------------------------------------------------
# Test histogram visual
#------------------------------------------------------------------------------

def test_histogram_empty(qtbot, canvas):
    hist = np.zeros((1, 0))
    _test_visual(qtbot, canvas, HistogramVisual(), hist=hist)


def test_histogram_0(qtbot, canvas_pz):
    hist = np.zeros((10,))
    _test_visual(qtbot, canvas_pz, HistogramVisual(), hist=hist)


def test_histogram_1(qtbot, canvas_pz):
    hist = np.random.rand(1, 10)
    _test_visual(qtbot, canvas_pz, HistogramVisual(), hist=hist)


def test_histogram_2(qtbot, canvas_pz):

    n_hists = 5
    hist = np.random.rand(n_hists, 21)

    # Histogram colors.
    c = np.random.uniform(.3, .6, size=(n_hists, 4))
    c[:, 3] = 1

    _test_visual(
        qtbot, canvas_pz, HistogramVisual(), hist=hist, color=c, ylim=2 * np.ones(n_hists))


def test_histogram_3(qtbot, canvas_pz):
    hist = np.random.rand(1, 100)
    visual = HistogramVisual()
    visual.transforms.add(Rotate())
    _test_visual(qtbot, canvas_pz, visual, hist=hist)


#------------------------------------------------------------------------------
# Test image visual
#------------------------------------------------------------------------------

def test_image_empty(qtbot, canvas):
    image = np.zeros((0, 0, 4))
    _test_visual(qtbot, canvas, ImageVisual(), image=image)


def test_image_1(qtbot, canvas):
    image = np.zeros((2, 2, 4))
    image[0, 0, 0] = 1
    image[1, 1, 1] = 2
    image[..., 3] = 1
    _test_visual(qtbot, canvas, ImageVisual(), image=image)


def test_image_2(qtbot, canvas):
    n = 100
    _test_visual(
        qtbot, canvas, ImageVisual(),
        image=np.random.uniform(low=.5, high=.9, size=(n, n, 4)))


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
    _test_visual(qtbot, canvas_pz, LineVisual(), pos=pos, color=color, data_bounds=[-1, -1, 1, 1])


#------------------------------------------------------------------------------
# Test line agg geom
#------------------------------------------------------------------------------

def test_line_agg_geom_0(qtbot, canvas_pz):
    n = 1024
    T = np.linspace(0, 10 * 2 * np.pi, n)
    R = np.linspace(0, .5, n)
    P = np.zeros((n, 2), dtype=np.float64)
    P[:, 0] = np.cos(T) * R
    P[:, 1] = np.sin(T) * R
    P = range_transform([NDC], [[0, 0, 1034, 1034]], P)

    color = np.random.uniform(.5, .9, 4)
    _test_visual(
        qtbot, canvas_pz, LineAggGeomVisual(), pos=P, color=color)


#------------------------------------------------------------------------------
# Test plot agg
#------------------------------------------------------------------------------

def test_plot_agg_empty(qtbot, canvas_pz):
    _test_visual(
        qtbot, canvas_pz, PlotAggVisual(), y=[])


def test_plot_agg_1(qtbot, canvas_pz):
    t = np.linspace(-np.pi, np.pi, 8)
    t = t[:-1]
    x = .5 * np.cos(t)
    y = .5 * np.sin(t)

    _test_visual(
        qtbot, canvas_pz, PlotAggVisual(closed=True), x=x, y=y, data_bounds='auto')


def test_plot_agg_2(qtbot, canvas_pz):
    n_signals = 100
    n_samples = 1000

    x = np.linspace(-1., 1., n_samples)
    y = np.sin(10 * x) * 0.1

    x = np.tile(x, (n_signals, 1))
    y = np.tile(y, (n_signals, 1))
    y -= np.linspace(-1, 1, n_signals)[:, np.newaxis]

    color = np.random.uniform(low=.5, high=.9, size=(n_signals, 4))
    depth = np.random.uniform(low=0, high=1, size=n_signals)
    masks = np.random.uniform(low=0, high=1, size=n_signals)

    _test_visual(
        qtbot, canvas_pz, PlotAggVisual(), x=x, y=y, color=color,
        depth=depth, masks=masks, data_bounds=NDC)


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
    _test_visual(qtbot, canvas_pz, TextVisual(), text='hello')


def test_text_1(qtbot, canvas_pz):
    text = '0123456789'
    text = [text[:n] for n in range(1, 11)]

    pos = np.c_[np.linspace(-.5, .5, 10), np.linspace(-.5, .5, 10)]
    color = np.ones((10, 4))
    color[:, 2] = 0

    _test_visual(
        qtbot, canvas_pz, TextVisual(font_size=32), pos=pos, text=text, color=color)


def test_text_2(qtbot, canvas_pz):
    c = canvas_pz

    text = ['12345'] * 5
    pos = [[0, 0], [-.5, +.5], [+.5, +.5], [-.5, -.5], [+.5, -.5]]
    anchor = [[0, 0], [-1, +1], [+1, +1], [-1, -1], [+1, -1]]

    v = TextVisual()
    c.add_visual(v)
    v.set_data(pos=pos, text=text, anchor=anchor, data_bounds=NDC)

    v = ScatterVisual()
    c.add_visual(v)
    v.set_data(pos=pos, data_bounds=None)

    v.set_marker_size(10)
    v.set_color(np.random.uniform(low=.5, high=.9, size=(v.n_vertices, 4)))

    c.show()
    qtbot.waitForWindowShown(c)

    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    c.close()


def test_text_3(qtbot, canvas_pz):
    text = '0123456789'
    text = [text] * 10

    _test_visual(
        qtbot, canvas_pz, TextVisual(color=(1, 1, 0, 1)), pos=[(0, 0)] * 10, text=text,
        anchor=[(1, -1 - 2 * i) for i in range(5)] + [(-1 - 2 * i, 1) for i in range(5)])
