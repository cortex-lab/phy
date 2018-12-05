# -*- coding: utf-8 -*-

"""Test plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import fixture

from ..plot import PlotCanvas
from ..transform import NDC
from ..utils import _get_linear_x
from .. import visuals


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def x():
    return .25 * np.random.randn(1000)


@fixture
def y():
    return .25 * np.random.randn(1000)


#------------------------------------------------------------------------------
# Test plotting interface
#------------------------------------------------------------------------------

def test_plot_1(qtbot, x, y):
    c = PlotCanvas(with_panzoom=True)

    v = visuals.ScatterVisual()
    c.add_visual(v)
    v.set_data(x=x, y=y)

    c.show()
    qtbot.waitForWindowShown(c)
    # qtbot.stop()
    c.close()


#------------------------------------------------------------------------------
# Test visuals in PlotCanvas
#------------------------------------------------------------------------------

def test_plot_grid(qtbot, x, y):
    c = PlotCanvas(layout='grid', shape=(2, 3))

    c[0, 0].add(visuals.PlotVisual(), x=x, y=y)
    c[0, 1].add(visuals.HistogramVisual(), 5 + x)
    c[0, 2].add(visuals.ScatterVisual(), x, y, color=np.random.uniform(.5, .8, size=(1000, 4)))

    c[1, 0].add(visuals.LineVisual(), pos=[-1, -5, +1, -.5])
    c[1, 1].add(visuals.TextVisual(), pos=(0, 0), text='Hello world!', anchor=(0., 0.))

    # Multiple scatters in the same subplot.
    c[1, 2].add(visuals.ScatterVisual(marker='asterisk'),
                x[2::6], y[2::6], color=(0, 1, 0, .25), size=20)
    c[1, 2].add(visuals.ScatterVisual(marker='heart'),
                x[::5], y[::5], color=(1, 0, 0, .35), size=50)
    c[1, 2].add(visuals.ScatterVisual(marker='heart'),
                x[1::3], y[1::3], color=(1, 0, 1, .35), size=30)

    c.show()
    qtbot.waitForWindowShown(c)
    # qtbot.stop()
    c.close()


def test_plot_stacked(qtbot):
    c = PlotCanvas(layout='stacked', n_plots=3)
    t = _get_linear_x(1, 1000).ravel()

    c[0].add(visuals.ScatterVisual(), pos=np.random.rand(100, 2))

    c[1].add(visuals.HistogramVisual(), np.random.rand(5, 10),
             color=np.random.uniform(.4, .9, size=(5, 4)))

    c[2].add(visuals.PlotVisual(), t, np.sin(20 * t), color=(1, 0, 0, 1))

    c.show()
    qtbot.waitForWindowShown(c)
    # qtbot.stop()
    c.close()


def test_plot_boxed(qtbot):
    n = 3
    b = np.zeros((n, 4))
    b[:, 0] = b[:, 1] = np.linspace(-1., 1. - 2. / 3., n)
    b[:, 2] = b[:, 3] = np.linspace(-1. + 2. / 3., 1., n)
    c = PlotCanvas(layout='boxed', box_bounds=b)

    t = _get_linear_x(1, 1000).ravel()
    c[0].add(visuals.ScatterVisual(), pos=np.random.rand(100, 2))
    c[1].add(visuals.PlotVisual(), t, np.sin(20 * t), color=(1, 0, 0, 1))
    c[2].add(visuals.HistogramVisual(), np.random.rand(5, 10),
             color=np.random.uniform(.4, .9, size=(5, 4)))

    c.show()
    qtbot.waitForWindowShown(c)
    # qtbot.stop()
    c.close()


#------------------------------------------------------------------------------
# Test lasso
#------------------------------------------------------------------------------

def _test_lasso_grid(qtbot):
    view = PlotCanvas(layout='grid', shape=(1, 2), enable_lasso=True)
    x, y = np.meshgrid(np.linspace(-1., 1., 20), np.linspace(-1., 1., 20))
    x, y = x.ravel(), y.ravel()
    view[0, 1].scatter(x, y, data_bounds=NDC)

    l = view.lasso
    ev = None
    # TODO
    return

    # Square selection in the left panel.
    ev.mouse_press(pos=(100, 100), button=1, modifiers=('Control',))
    assert l.box == (0, 0)
    ev.mouse_press(pos=(200, 100), button=1, modifiers=('Control',))
    ev.mouse_press(pos=(200, 200), button=1, modifiers=('Control',))
    ev.mouse_press(pos=(100, 200), button=1, modifiers=('Control',))
    assert l.box == (0, 0)

    # Clear.
    ev.mouse_press(pos=(100, 200), button=2, modifiers=('Control',))
    assert l.box is None

    # Square selection in the right panel.
    ev.mouse_press(pos=(500, 100), button=1, modifiers=('Control',))
    assert l.box == (0, 1)
    ev.mouse_press(pos=(700, 100), button=1, modifiers=('Control',))
    ev.mouse_press(pos=(700, 300), button=1, modifiers=('Control',))
    ev.mouse_press(pos=(500, 300), button=1, modifiers=('Control',))
    assert l.box == (0, 1)

    ind = l.in_polygon(np.c_[x, y])
    view[0, 1].scatter(x[ind], y[ind], color=(1., 0., 0., 1.),
                       data_bounds=NDC)
