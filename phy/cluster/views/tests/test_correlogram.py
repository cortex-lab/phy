# -*- coding: utf-8 -*-

"""Test correlogram view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from .conftest import _select_clusters


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot, gui):
    v = gui.controller.add_correlogram_view(gui)
    _select_clusters(gui)

    v.toggle_normalization()

    v.set_bin(1)
    v.set_window(100)

    # qtbot.stop()
    gui.close()
