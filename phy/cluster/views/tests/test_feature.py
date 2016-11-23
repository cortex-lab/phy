# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from .conftest import _select_clusters


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot, gui):
    v = gui.controller.add_feature_view(gui)
    _select_clusters(gui)

    assert v.feature_scaling == .5
    v.add_attribute('sine',
                    np.sin(np.linspace(-10., 10., gui.controller.n_spikes)))

    v.increase()
    v.decrease()

    v.on_channel_click(channel_idx=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # qtbot.stop()
    gui.close()
