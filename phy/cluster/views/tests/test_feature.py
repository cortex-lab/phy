# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.gui import GUI
from phy.io.mock import (artificial_features,
                         )
from phy.utils import Bunch

from ..feature import FeatureView


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot):
    nc = 5
    ns = 50
    features = artificial_features(ns, nc, 4)

    def get_features(cluster_id=None, n_spikes=None, channel_ids=None):
        return Bunch(data=features,
                     channel_ids=np.arange(nc),
                     )

    v = FeatureView(features=get_features,
                    )
    gui = GUI()
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    qtbot.waitForWindowShown(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    qtbot.stop()
    return

    v.increase()
    v.decrease()

    v.on_channel_click(channel_idx=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # qtbot.stop()
    gui.close()
