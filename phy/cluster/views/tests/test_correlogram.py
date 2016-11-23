# -*- coding: utf-8 -*-

"""Test correlogram view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.gui import GUI
from phy.io.mock import (artificial_correlograms,
                         )

from ..correlogram import CorrelogramView


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot):

    nc = 5
    ns = 50

    def get_correlograms(cluster_id, bin_size, window_size):
        return artificial_correlograms(nc, ns)

    v = CorrelogramView(correlograms=get_correlograms,
                        sample_rate=100.,
                        )
    gui = GUI()
    gui.show()
    v.attach(gui)

    # qtbot.waitForWindowShown(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    v.toggle_normalization()

    v.set_bin(1)
    v.set_window(100)

    # qtbot.stop()
    gui.close()
