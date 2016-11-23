# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.gui import GUI
from phy.io.mock import (artificial_features,
                         artificial_spike_samples,
                         )
from phy.utils import Bunch

from ..feature import FeatureView


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot):
    nc = 5
    ns = 50

    def get_features(cluster_id):
        return Bunch(data=artificial_features(ns, nc, 4),
                     )

    v = FeatureView(features=get_features,
                    n_channels=nc,
                    spike_times=artificial_spike_samples(ns) / 1000.,
                    )
    gui = GUI()
    gui.show()
    v.attach(gui)

    # qtbot.waitForWindowShown(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    v.add_attribute('sine', np.sin(np.linspace(-10., 10., ns)))

    v.increase()
    v.decrease()

    v.on_channel_click(channel_idx=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # qtbot.stop()
    gui.close()
