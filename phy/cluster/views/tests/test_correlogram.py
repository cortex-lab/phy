# -*- coding: utf-8 -*-

"""Test correlogram view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.io.mock import (artificial_correlograms,
                         )

from ..correlogram import CorrelogramView


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot):

    def get_correlograms(cluster_ids, bin_size, window_size):
        return artificial_correlograms(len(cluster_ids), int(window_size / bin_size))

    v = CorrelogramView(correlograms=get_correlograms,
                        sample_rate=100.,
                        )
    v.canvas.show()
    qtbot.waitForWindowShown(v.canvas)

    v.on_select([])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.toggle_normalization(True)

    v.set_bin(1)
    v.set_window(100)

    # qtbot.stop()
    v.canvas.close()
