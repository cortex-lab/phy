# -*- coding: utf-8 -*-

"""Test correlogram view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.mock import artificial_correlograms

from ..correlogram import CorrelogramView


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot, gui):

    def get_correlograms(cluster_ids, bin_size, window_size):
        return artificial_correlograms(len(cluster_ids), int(window_size / bin_size))

    def get_firing_rate(cluster_ids, bin_size):
        return .5 * np.ones((len(cluster_ids), len(cluster_ids)))

    v = CorrelogramView(correlograms=get_correlograms,
                        firing_rate=get_firing_rate,
                        sample_rate=100.,
                        )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select([])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.toggle_normalization(True)

    v.set_bin(1)
    v.set_window(100)

    v.set_state(v.state)

    # qtbot.stop()
    v.close()
