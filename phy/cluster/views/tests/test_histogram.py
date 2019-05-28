# -*- coding: utf-8 -*-

"""Test Histogram view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import Bunch
from ..histogram import HistogramView


#------------------------------------------------------------------------------
# Test Histogram view
#------------------------------------------------------------------------------

def test_histogram_view_0(qtbot, gui):
    v = HistogramView(
        cluster_stat=lambda cluster_id: Bunch(
            histogram=np.random.uniform(low=0, high=10, size=10),
            plot=np.random.uniform(low=5, high=10, size=100),
            text='this is:\ncluster %d' % cluster_id,
            data_bounds=(0, 0, 27, 10),
        )
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[2, 3, 5])
    # qtbot.stop()
    v.close()
