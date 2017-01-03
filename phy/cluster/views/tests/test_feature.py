# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import pytest

from phy.gui import GUI, GUIState
from phy.gui.qt import Qt, QPoint
from phy.io.array import _spikes_per_cluster
from phy.io.mock import (artificial_features,
                         artificial_spike_clusters,
                         )
from phy.utils import Bunch

from ..feature import FeatureView


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

@pytest.mark.parametrize('n_channels', [5, 1])
def test_feature_view(qtbot, tempdir, n_channels):
    nc = n_channels
    ns = 500
    features = artificial_features(ns, nc, 4)
    spike_clusters = artificial_spike_clusters(ns, 4)
    spike_times = np.linspace(0., 1., ns)
    spc = _spikes_per_cluster(spike_clusters)

    def get_spike_ids(cluster_id):
        return (spc[cluster_id] if cluster_id is not None else np.arange(ns))

    def get_features(cluster_id=None, channel_ids=None, spike_ids=None,
                     load_all=None):
        if load_all:
            spike_ids = spc[cluster_id]
        else:
            spike_ids = get_spike_ids(cluster_id)
        return Bunch(data=features[spike_ids],
                     spike_ids=spike_ids,
                     masks=np.random.rand(ns, nc),
                     channel_ids=(channel_ids
                                  if channel_ids is not None
                                  else np.arange(nc)[::-1]),
                     )

    def get_time(cluster_id=None, load_all=None):
        return Bunch(data=spike_times[get_spike_ids(cluster_id)],
                     lim=(0., 1.),
                     )

    v = FeatureView(features=get_features,
                    attributes={'time': get_time},
                    )

    v.set_state(GUIState(scaling=None))

    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    gui.emit('select', [0, 2])
    qtbot.wait(10)

    v.increase()
    v.decrease()

    v.on_channel_click(channel_id=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # Split without selection.
    spike_ids = v.on_request_split()
    assert len(spike_ids) == 0

    # Draw a lasso.
    def _click(x, y):
        qtbot.mouseClick(v.native, Qt.LeftButton, pos=QPoint(x, y),
                         modifier=Qt.ControlModifier)

    _click(10, 10)
    _click(10, 100)
    _click(100, 100)
    _click(100, 10)

    # Split lassoed points.
    spike_ids = v.on_request_split()
    assert len(spike_ids) > 0

    # qtbot.stop()
    gui.close()
