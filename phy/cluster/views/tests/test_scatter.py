# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import pytest

from phy.gui import GUI
from phy.utils import Bunch
from ..scatter import ScatterView, ScatterViewMatplotlib


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

@pytest.mark.parametrize('view_class', (ScatterView, ScatterViewMatplotlib))
def test_scatter_view(qtbot, tempdir, view_class):
    n = 1000
    v = view_class(
        coords=lambda c: Bunch(
            x=np.random.randn(n),
            y=np.random.randn(n),
            data_bounds=None,
        )
    )
    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    # qtbot.stop()
    gui.close()
