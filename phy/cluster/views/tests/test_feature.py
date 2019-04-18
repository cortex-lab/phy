# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import pytest

from phy.io.array import _spikes_per_cluster
from phy.io.mock import (artificial_features,
                         artificial_spike_clusters,
                         )
from phy.plot.tests import mouse_click
from phy.utils import Bunch
from phy.utils.testing import _in_travis

from ..feature import FeatureView


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

@pytest.mark.parametrize('n_channels', [5, 1])
def test_feature_view(qtbot, gui, n_channels):
    nc = n_channels
    ns = 10000
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
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.increase()
    v.decrease()

    v.on_channel_click(channel_id=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection(True)

    # Split without selection.
    spike_ids = v.on_request_split()
    assert len(spike_ids) == 0

    a, b = 50, 200
    mouse_click(qtbot, v.canvas, (a, a), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (a, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, a), modifiers=('Control',))

    # Split lassoed points.
    if not _in_travis():
        spike_ids = v.on_request_split()
        assert len(spike_ids) > 0

    v.set_state(v.state)

    # qtbot.stop()
    v.close()
