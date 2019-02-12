# -*- coding: utf-8 -*-

"""Test axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac

from ..axes import AxisLocator, Axes


#------------------------------------------------------------------------------
# Tests axes
#------------------------------------------------------------------------------

def test_locator_1():
    l = AxisLocator('auto')

    l.set_view_bounds((0., 0., 1., 1.))
    ae(l.xticks, l.yticks)
    ae(l.xticks, np.linspace(-1., 2., 7))

    l.set_view_bounds((.101, -201, .201, -101))
    ac(l.xticks, np.linspace(0, .35, 8))
    ae(l.yticks, np.linspace(-350., 0., 8))


def test_axes_1(qtbot, canvas_pz):
    c = canvas_pz

    g = Axes(data_bounds=(0, -10, 1000, 10))
    g.attach(c)

    c.show()
    qtbot.waitForWindowShown(c)

    c.panzoom.zoom = 4
    #c.panzoom.zoom = 8
    #c.panzoom.pan = (3, 3)

    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    c.close()
