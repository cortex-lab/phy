# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.gui import GUI
from phy.io.array import _spikes_per_cluster
from phy.io.mock import (artificial_features,
                         artificial_spike_clusters,
                         )
from phy.utils import Bunch

from ..feature import FeatureView


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot, tempdir):
    nc = 5
    ns = 500
    features = artificial_features(ns, nc, 4)
    spike_clusters = artificial_spike_clusters(ns, 4)
    spike_times = np.linspace(0., 1., ns)
    spc = _spikes_per_cluster(spike_clusters)

    def get_spike_ids(cluster_id):
        return (spc[cluster_id] if cluster_id is not None else np.arange(ns))

    def get_features(cluster_id=None, n_spikes=None, channel_ids=None):
        return Bunch(data=features[get_spike_ids(cluster_id)],
                     channel_ids=np.arange(nc)[::-1],
                     )

    def get_time(cluster_id=None):
        return Bunch(data=spike_times[get_spike_ids(cluster_id)],
                     lim=(0., 1.),
                     )

    v = FeatureView(features=get_features,
                    attributes={'time': get_time},
                    )
    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    v.increase()
    v.decrease()

    v.on_channel_click(channel_idx=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # qtbot.stop()
    gui.close()
