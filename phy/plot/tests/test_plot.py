# -*- coding: utf-8 -*-

"""Test plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..panzoom import PanZoom
from ..plot import BaseView, GridView, BoxedView, StackedView
from ..utils import _get_linear_x


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _show(qtbot, view, stop=False):
    view.build()
    view.show()
    qtbot.waitForWindowShown(view.native)
    if stop:  # pragma: no cover
        qtbot.stop()
    view.close()


#------------------------------------------------------------------------------
# Test plotting interface
#------------------------------------------------------------------------------

def test_base_view(qtbot):
    view = BaseView(keys='interactive')
    n = 1000

    x = np.random.randn(n)
    y = np.random.randn(n)

    view.scatter(x, y)
    _show(qtbot, view)


def test_grid_scatter(qtbot):
    view = GridView((2, 3))
    n = 100

    assert isinstance(view.panzoom, PanZoom)

    x = np.random.randn(n)
    y = np.random.randn(n)

    view[0, 1].scatter(x, y)
    view[0, 2].scatter(x, y, color=np.random.uniform(.5, .8, size=(n, 4)))

    view[1, 0].scatter(x, y, size=np.random.uniform(5, 20, size=n))
    view[1, 1]

    # Multiple scatters in the same subplot.
    view[1, 2].scatter(x[2::6], y[2::6], marker='asterisk',
                       color=(0, 1, 0, .25), size=20)
    view[1, 2].scatter(x[::5], y[::5], marker='heart',
                       color=(1, 0, 0, .35), size=50)
    view[1, 2].scatter(x[1::3], y[1::3], marker='heart',
                       color=(1, 0, 1, .35), size=30)

    _show(qtbot, view)


def test_grid_plot(qtbot):
    view = GridView((1, 2))
    n_plots, n_samples = 10, 50

    x = _get_linear_x(n_plots, n_samples)
    y = np.random.randn(n_plots, n_samples)

    view[0, 0].plot(x, y)
    view[0, 1].plot(x, y, color=np.random.uniform(.5, .8, size=(n_plots, 4)))

    _show(qtbot, view)


def test_grid_hist(qtbot):
    view = GridView((3, 3))

    hist = np.random.rand(3, 3, 20)

    for i in range(3):
        for j in range(3):
            view[i, j].hist(hist[i, j, :],
                            color=np.random.uniform(.5, .8, size=4))

    _show(qtbot, view)


def test_grid_complete(qtbot):
    view = GridView((2, 2))
    t = _get_linear_x(1, 1000).ravel()

    view[0, 0].scatter(*np.random.randn(2, 100))
    view[0, 1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))

    view[1, 1].hist(np.random.rand(5, 10),
                    color=np.random.uniform(.4, .9, size=(5, 4)))

    _show(qtbot, view)


def test_stacked_complete(qtbot):
    view = StackedView(3)

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
    view = BoxedView(b)

    t = _get_linear_x(1, 1000).ravel()
    view[0].scatter(*np.random.randn(2, 100))
    view[1].plot(t, np.sin(20 * t), color=(1, 0, 0, 1))
    view[2].hist(np.random.rand(5, 10),
                 color=np.random.uniform(.4, .9, size=(5, 4)))

    _show(qtbot, view)
