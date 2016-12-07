# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.gui import GUI
from phy.utils import Bunch
from ..scatter import ScatterView


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

def test_scatter_view(qtbot, tempdir):
    n = 1000
    v = ScatterView(coords=lambda c: Bunch(x=np.random.randn(n),
                                           y=np.random.randn(n),
                                           data_bounds=None,
                                           )
                    )
    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    # qtbot.stop()
    gui.close()
