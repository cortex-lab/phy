# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.utils import Bunch
from ..scatter import ScatterView
from .conftest import _select_clusters


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

def test_scatter_view(qtbot, gui):
    n = 1000
    v = ScatterView(coords=lambda c: [Bunch(x=np.random.randn(n),
                                            y=np.random.randn(n),
                                            spike_ids=np.arange(n),
                                            spike_clusters=np.ones(n).
                                            astype(np.int32) * c[0],
                                            )] if 2 not in c else None,
                    # data_bounds=[-3, -3, 3, 3],
                    )
    v.attach(gui)

    _select_clusters(gui)

    # qtbot.stop()
    gui.close()
