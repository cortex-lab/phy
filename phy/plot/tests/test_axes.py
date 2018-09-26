# -*- coding: utf-8 -*-

"""Test axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac

from ..axes import AxisLocator, Axes


#------------------------------------------------------------------------------
# Tests axes
#------------------------------------------------------------------------------

def test_locator_1():
    l = AxisLocator('auto')

    xticks, yticks = l.get_ticks((0, 0, 1, 1))
    ae(xticks, yticks)
    ae(xticks, np.linspace(-1., 2., 7))

    xticks, yticks = l.get_ticks((.101, -201, .201, -101))
    ac(xticks, np.linspace(0, .35, 8))
    ae(yticks, np.linspace(-350., 0., 8))


def test_locator_2():
    l = AxisLocator()
    assert l.format(0) == '0'
    assert l.format(0.1) == '1e-1'


def test_axes_1(qtbot, canvas_pz):
    c = canvas_pz

    g = Axes()
    g.attach(c)

    c.show()
    qtbot.waitForWindowShown(c.native)

    c.panzoom.zoom = 2
    c.panzoom.zoom = 8
    c.panzoom.pan = (3, 3)

    # qtbot.stop()
    c.close()
