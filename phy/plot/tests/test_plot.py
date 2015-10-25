# -*- coding: utf-8 -*-

"""Test plotting interface."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..plot import GridView


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

def test_subplot_view(qtbot):
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
