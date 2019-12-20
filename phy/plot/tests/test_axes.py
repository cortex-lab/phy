# -*- coding: utf-8 -*-

"""Test axes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

from ..axes import Axes


#------------------------------------------------------------------------------
# Tests axes
#------------------------------------------------------------------------------

def test_axes_1(qtbot, canvas_pz):
    c = canvas_pz

    db = (0, -10, 1000, 10)
    g = Axes(data_bounds=db)
    g.attach(c)

    c.show()
    qtbot.waitForWindowShown(c)

    c.panzoom.zoom = 4
    c.panzoom.zoom = 8
    c.panzoom.pan = (3, 3)
    g.reset_data_bounds(db)

    g._update_zoom(c.panzoom.zoom)
    g._update_pan(c.panzoom.pan)

    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    c.close()
