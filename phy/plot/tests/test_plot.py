# -*- coding: utf-8 -*-

"""Test plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from pytest import fixture, raises

from phy.gui import GUI
from ..plot import PlotCanvas, PlotCanvasMpl
from ..utils import _get_linear_x


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

@fixture(params=[True, False])
def canvas(request, qtbot):
    c = PlotCanvas() if request.param else PlotCanvasMpl()
    yield c
    c.show()
    qtbot.waitForWindowShown(c.canvas)
    if os.environ.get('PHY_TEST_STOP', None):
        qtbot.stop()
    c.close()


def test_plot_0(qtbot, x, y):
    c = PlotCanvas()
    c.has_axes = True
    c.has_lasso = True
    c.scatter(x=x, y=y)
    c.show()
    qtbot.waitForWindowShown(c.canvas)
    #c._enable()
    c.close()


def test_plot_1(canvas, x, y):
    c = canvas
    c.scatter(x=x, y=y)
    c.enable_axes()
    c.enable_lasso()


def test_plot_grid(canvas, x, y):
    c = canvas
    c.set_layout('grid', shape=(2, 3))

    c[0, 0].plot(x=x, y=y)
    c[0, 1].hist(5 + x[::10])
    c[0, 2].scatter(x, y, color=np.random.uniform(.5, .8, size=(1000, 4)))

    c[1, 0].lines(pos=[-1, -.5, +1, -.5])
    c[1, 1].text(pos=(0, 0), text='Hello world!', anchor=(0., 0.))
    c[1, 1].polygon(pos=np.random.rand(5, 2))

    # Multiple scatters in the same subplot.
    c[1, 2].scatter(x[2::6], y[2::6], color=(0, 1, 0, .25), size=20, marker='asterisk')
    c[1, 2].scatter(x[::5], y[::5], color=(1, 0, 0, .35), size=50, marker='heart')
    c[1, 2].scatter(x[1::3], y[1::3], color=(1, 0, 1, .35), size=30, marker='heart')


def test_plot_stacked(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    c = canvas
    c.set_layout('stacked', n_plots=3)

    t = _get_linear_x(1, 1000).ravel()
    c[0].scatter(pos=np.random.rand(100, 2))

    c[1].hist(np.random.rand(5, 10), color=np.random.uniform(.4, .9, size=(5, 4)))

    c[2].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))


def test_plot_boxed(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    c = canvas

    n = 3
    b = np.zeros((n, 4))
    b[:, 0] = b[:, 1] = np.linspace(-1., 1. - 2. / 3., n)
    b[:, 2] = b[:, 3] = np.linspace(-1. + 2. / 3., 1., n)
    c.set_layout('boxed', box_bounds=b)

    t = _get_linear_x(1, 1000).ravel()
    c[0].scatter(pos=np.random.rand(100, 2))
    c[1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))
    c[2].hist(np.random.rand(5, 10),
              color=np.random.uniform(.4, .9, size=(5, 4)))


def test_plot_uplot(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    x, y = .25 * np.random.randn(2, 1000)
    canvas.uplot(x=x, y=y)


def test_plot_uscatter(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    x, y = .25 * np.random.randn(2, 1000)
    canvas.uscatter(x=x, y=y)


def test_plot_uscatter_batch_fail(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    x, y = .25 * np.random.randn(2, 100)
    with raises(TypeError):  # color cannot be passed to batch
        canvas.uscatter_batch(x=x, y=y, color=(1, 0, 0, 1))


def test_plot_batch_1(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    canvas.set_layout('grid', shape=(1, 2))
    x, y = .25 * np.random.randn(2, 10)
    canvas[0, 0].plot_batch(x=x, y=y)
    canvas[0, 1].plot_batch(x=y, y=x)
    canvas.plot()


def test_plot_uscatter_batch_1(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    canvas.set_layout('grid', shape=(1, 2))
    x, y = .25 * np.random.randn(2, 10)
    canvas[0, 0].uscatter_batch(x=x, y=y)
    canvas[0, 1].uscatter_batch(x=y, y=x)
    canvas.uscatter(color=(1, 1, 0, 1))


def test_plot_lines_batch_1(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    canvas.set_layout('grid', shape=(1, 2))
    canvas[0, 0].lines_batch(pos=[0., 0., 1., 1.])
    canvas[0, 1].lines_batch(pos=[-1., 1., 1., -1.])
    canvas.lines()


def test_plot_hist_batch_1(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    canvas.set_layout('grid', shape=(1, 2))
    canvas[0, 0].hist_batch(hist=[[3., 4., 1., 2.]], color=(1, 0, 0, 1))
    canvas[0, 1].hist_batch(hist=[[0., 1., 0., 2.], [1., 0., 3., 0.]], color=(0, 1, 0, 1))
    canvas.hist()


def test_plot_text_batch_1(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    canvas.set_layout('grid', shape=(1, 2))
    canvas[0, 0].text(pos=(0, 0), text='12', anchor=(0., 0.))
    canvas[0, 1].text(pos=(0, 0), text='345', anchor=(0., 0.))


def test_plot_text_batch_2(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    canvas.set_layout('grid', shape=(1, 2))
    canvas[0, 0].text_batch(pos=(0, 0), text='12', anchor=(0., 0.))
    canvas[0, 1].text_batch(pos=(0, 0), text='345', anchor=(0., 0.))
    canvas.text()


#------------------------------------------------------------------------------
# Test matplotlib plotting
#------------------------------------------------------------------------------

def test_plot_mpl_1(qtbot):
    gui = GUI()
    c = PlotCanvasMpl()

    c.clear()
    c.attach(gui)

    c.show()
    qtbot.waitForWindowShown(c.canvas)
    if os.environ.get('PHY_TEST_STOP', None):
        qtbot.stop()
    c.close()
