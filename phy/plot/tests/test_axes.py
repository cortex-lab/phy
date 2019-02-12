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
