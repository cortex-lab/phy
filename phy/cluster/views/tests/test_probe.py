# -*- coding: utf-8 -*-

"""Test probe view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.electrode.mea import staggered_positions
from phy.utils import emit

from ..probe import ProbeView


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_probe_view(qtbot):

    n = 50
    positions = staggered_positions(n)
    best_channels = lambda cluster_id: range(1, 9, 2)

    v = ProbeView(positions=positions,
                  best_channels=best_channels,
                  )
    v.show()
    qtbot.addWidget(v)

    class Supervisor(object):
        pass

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    emit('select', Supervisor(), cluster_ids=[0, 2])

    # qtbot.stop()
    v.close()
