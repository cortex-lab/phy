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
# Test visuals in grid
#------------------------------------------------------------------------------

def test_grid_scatter(qtbot, x, y):
    c = PlotCanvas(layout='grid', shape=(2, 3))

    c[0, 1].add(visuals.ScatterVisual(), x=x, y=y)
    c[1, 0].add(visuals.ScatterVisual(), x, y, size=np.random.uniform(5, 20, size=1000))
    c[0, 2].add(visuals.ScatterVisual(), x, y, color=np.random.uniform(.5, .8, size=(1000, 4)))

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


def test_grid_plot(qtbot):
    view = View(layout='grid', shape=(1, 2))
    n_plots, n_samples = 5, 50

    x = _get_linear_x(n_plots, n_samples)
    y = np.random.randn(n_plots, n_samples)

    view[0, 0].plot(x, y)
    view[0, 1].plot(x, y, color=np.random.uniform(.5, .8, size=(n_plots, 4)))

    _show(qtbot, view)


def test_grid_plot_uniform(qtbot):
    view = View(layout='grid', shape=(1, 2))
    n_plots, n_samples = 5, 50

    x = _get_linear_x(n_plots, n_samples)
    y = np.random.randn(n_plots, n_samples)

    view[0, 0].uplot(x, y)
    view[0, 1].uplot(x, y, color=(1., 1., 0., .5))

    _show(qtbot, view)


def test_grid_hist(qtbot):
    view = View(layout='grid', shape=(3, 3))

    hist = np.random.rand(3, 3, 20)

    for i in range(3):
        for j in range(3):
            view[i, j].hist(hist[i, j, :],
                            color=np.random.uniform(.5, .8, size=4))

    _show(qtbot, view)


def test_grid_lines(qtbot):
    view = View(layout='grid', shape=(1, 2))

    view[0, 0].lines(pos=[-1, -.5, +1, -.5])
    view[0, 1].lines(pos=[-1, +.5, +1, +.5])

    _show(qtbot, view)


def test_grid_text(qtbot):
    view = View(layout='grid', shape=(2, 1))

    view[0, 0].text(pos=(0, 0), text='Hello world!', anchor=(0., 0.))
    view[1, 0].text(pos=[[-.5, 0], [+.5, 0]], text=['|', ':)'])

    _show(qtbot, view)


def test_grid_complete(qtbot):
    view = View(layout='grid', shape=(2, 2))
    t = _get_linear_x(1, 1000).ravel()

    view[0, 0].scatter(*np.random.randn(2, 100))
    view[0, 1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))

    view[1, 1].hist(np.random.rand(5, 10),
                    color=np.random.uniform(.4, .9, size=(5, 4)))

    _show(qtbot, view)


#------------------------------------------------------------------------------
# Test other interact
#------------------------------------------------------------------------------

def test_stacked_complete(qtbot):
    view = View(layout='stacked', n_plots=3)

    t = _get_linear_x(1, 1000).ravel()
    view[0].scatter(*np.random.randn(2, 100))

    # Different types of visuals in the same subplot.
    view[1].hist(np.random.rand(5, 10),
                 color=np.random.uniform(.4, .9, size=(5, 4)))
    view[1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))

    # TODO
    # v = view[2].plot(t[::2], np.sin(20 * t[::2]), color=(1, 0, 0, 1))
    # v.update(color=(0, 1, 0, 1))

    _show(qtbot, view)


def test_boxed_complete(qtbot):
    n = 3
    b = np.zeros((n, 4))
    b[:, 0] = b[:, 1] = np.linspace(-1., 1. - 2. / 3., n)
    b[:, 2] = b[:, 3] = np.linspace(-1. + 2. / 3., 1., n)
    view = View(layout='boxed', box_bounds=b)

    t = _get_linear_x(1, 1000).ravel()
    view[0].scatter(*np.random.randn(2, 100))
    view[1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))
    view[2].hist(np.random.rand(5, 10),
                 color=np.random.uniform(.4, .9, size=(5, 4)))

    _show(qtbot, view)


#------------------------------------------------------------------------------
# Test lasso
#------------------------------------------------------------------------------

def _test_lasso_simple(qtbot):
    view = View(enable_lasso=True)
    n = 1000

    x = np.random.randn(n)
    y = np.random.randn(n)

    view.scatter(x, y)

    l = view.lasso
    assert str(l)
    # TODO
    #ev.mouse_press(pos=(0, 0), button=1, modifiers=('Control',))
    l.add((+1, -1))
    l.add((+1, +1))
    l.add((-1, +1))
    assert l.count == 4
    assert l.polygon.shape == (4, 2)
    b = [[-1, -1], [+1, -1], [+1, +1], [-1, +1]]
    ae(l.in_polygon(b), [False, False, True, True])

    #ev.mouse_press(pos=(0, 0), button=2, modifiers=('Control',))
    assert l.count == 0

    _show(qtbot, view)


def _test_lasso_grid(qtbot):
    view = View(layout='grid', shape=(1, 2),
                enable_lasso=True)
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

    _show(qtbot, view)
