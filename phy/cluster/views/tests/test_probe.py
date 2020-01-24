# -*- coding: utf-8 -*-

"""Test probe view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils.geometry import staggered_positions
from phylib.utils import emit

from ..probe import ProbeView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_probe_view(qtbot, gui):

    n = 50
    positions = staggered_positions(n)
    positions = positions.astype(np.int32)
    best_channels = lambda cluster_id: range(1, 9, 2)

    v = ProbeView(positions=positions, best_channels=best_channels, dead_channels=(3, 7, 12))
    v.do_show_labels = False
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    class Supervisor(object):
        pass

    v.toggle_show_labels(True)
    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    emit('select', Supervisor(), cluster_ids=[0, 2])

    v.toggle_show_labels(False)

    _stop_and_close(qtbot, v)
