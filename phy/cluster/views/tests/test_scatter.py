# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.utilslib import Bunch
from ..scatter import ScatterView


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

def test_scatter_view(qtbot, gui):
    n = 1000
    v = ScatterView(
        coords=lambda c: Bunch(
            x=np.random.randn(n),
            y=np.random.randn(n),
            data_bounds=None,
        )
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.set_state(v.state)

    # qtbot.stop()
    v.close()
