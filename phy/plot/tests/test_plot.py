# -*- coding: utf-8 -*-

"""Test plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..plot import GridView
from ..visuals import _get_linear_x


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

def test_grid_scatter(qtbot):
    view = GridView(2, 3)
    n = 1000

    x = np.random.randn(n)
    y = np.random.randn(n)

    view[0, 1].scatter(x, y)
    view[0, 2].scatter(x, y, color=np.random.uniform(.5, .8, size=(n, 4)))

    view[1, 0].scatter(x, y, size=np.random.uniform(5, 20, size=n))
    view[1, 1]
    view[1, 2].scatter(x[::5], y[::5], marker='heart',
                       color=(1, 0, 0, .25), size=20)

    _show(qtbot, view)


def test_grid_plot(qtbot):
    view = GridView(1, 2)
    n_plots, n_samples = 10, 50

    x = _get_linear_x(n_plots, n_samples)
    y = np.random.randn(n_plots, n_samples)

    view[0, 0].plot(x, y)
    view[0, 1].plot(x, y, color=np.random.uniform(.5, .8, size=(n_plots, 4)))

    _show(qtbot, view)
