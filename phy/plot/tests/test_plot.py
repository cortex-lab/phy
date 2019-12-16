# -*- coding: utf-8 -*-

"""Test plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from pytest import fixture

from phy.gui import GUI
from ..plot import PlotCanvas, PlotCanvasMpl
from ..utils import get_linear_x
from ..visuals import PlotVisual, TextVisual


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

    t = get_linear_x(1, 1000).ravel()
    c[0].scatter(pos=np.random.rand(100, 2))

    c[1].hist(np.random.rand(5, 10), color=np.random.uniform(.4, .9, size=(5, 4)))

    c[2].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))


def test_plot_boxed(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return
    c = canvas

    n = 3
    b = np.zeros((n, 2))
    b[:, 0] = np.linspace(-1., 1., n)
    b[:, 1] = np.linspace(-1., 1., n)
    c.set_layout('boxed', box_pos=b)

    t = get_linear_x(1, 1000).ravel()
    c[0].scatter(pos=np.random.rand(100, 2))
    c[1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))
    c[2].hist(np.random.rand(5, 10), color=np.random.uniform(.4, .9, size=(5, 4)))


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


def test_plot_batch_1(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return

    visual = PlotVisual()

    for size in (3, 5):
        x = np.linspace(-1, 1, size)
        y = np.random.uniform(0, 1, size)
        visual.add_batch_data(x=x, y=y)

    canvas.add_visual(visual)


def test_plot_batch_2(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return

    canvas.set_layout('grid', shape=(2, 1))

    visual = PlotVisual()
    visual.reset_batch()

    for i, size in enumerate((3, 5)):
        x = np.linspace(-1, 1, size)
        y = np.random.uniform(-1, 1, size)
        visual.add_batch_data(x=x, y=y, box_index=(i, 0))

    canvas.add_visual(visual)


def test_plot_batch_3(qtbot, canvas):
    if isinstance(canvas, PlotCanvasMpl):
        # TODO: not implemented yet
        return

    canvas.set_layout('grid', shape=(2, 1))

    visual = TextVisual()
    canvas.add_visual(visual)

    visual.add_batch_data(
        pos=np.zeros((5, 2)), text=["a" * (i + 1) for i in range(5)],
        data_bounds=None, box_index=(0, 0))

    visual.add_batch_data(
        pos=np.zeros((7, 2)), text=["a" * (i + 1) for i in range(7)],
        data_bounds=(-1, -1, 1, 1), box_index=(0, 1))

    canvas.update_visual(visual)


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
