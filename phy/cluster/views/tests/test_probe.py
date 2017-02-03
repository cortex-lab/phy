# -*- coding: utf-8 -*-

"""Test probe view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.gui import GUI
from phy.electrode.mea import staggered_positions

from ..probe import ProbeView


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_probe_view(qtbot, tempdir):

    n = 50
    positions = staggered_positions(n)
    best_channels = lambda cluster_id: range(1, 9, 2)

    v = ProbeView(positions=positions,
                  best_channels=best_channels,
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
