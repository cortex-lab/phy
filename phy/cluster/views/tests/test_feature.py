# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import pytest

from phylib.io.array import _spikes_per_cluster
from phylib.io.mock import artificial_features, artificial_spike_clusters
from phylib.utils import Bunch, connect
from phy.plot.tests import mouse_click

from ..feature import FeatureView, _get_default_grid
from . import _stop_and_close


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

    def get_features(cluster_id=None, channel_ids=None, spike_ids=None, load_all=None):
        if load_all:
            spike_ids = spc[cluster_id]
        else:
            spike_ids = get_spike_ids(cluster_id)
        return Bunch(
            data=features[spike_ids],
            spike_ids=spike_ids,
            masks=np.random.rand(ns, nc),
            channel_ids=(channel_ids if channel_ids is not None else np.arange(nc)[::-1]),
        )

    def get_time(cluster_id=None, load_all=None):
        return Bunch(data=spike_times[get_spike_ids(cluster_id)], lim=(0., 1.))

    v = FeatureView(features=get_features, attributes={'time': get_time})
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.set_grid_dim(_get_default_grid())

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    assert v.status

    v.increase()
    v.decrease()

    v.increase_marker_size()
    v.decrease_marker_size()

    v.on_select_channel(channel_id=3, button='Left', key=None)
    v.on_select_channel(channel_id=3, button='Right', key=None)
    v.on_select_channel(channel_id=3, button='Right', key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection(True)

    # Test feature selection with Alt+click.
    _l = []

    @connect(sender=v)
    def on_select_feature(sender, dim=None, channel_id=None, pc=None):
        _l.append((dim, channel_id, pc))

    for i, j, dim_x, dim_y in v._iter_subplots():
        for k, button in enumerate(('Left', 'Right')):
            # Click on the center of every subplot.
            w, h = v.canvas.get_size()
            w, h = w / 4, h / 4
            x, y = w / 2, h / 2
            mouse_click(qtbot, v.canvas, (x + j * w, y + i * h), button=button, modifiers=('Alt',))
            assert _l[-1][0] == v.grid_dim[i][j].split(',')[k]

    # Split without selection.
    spike_ids = v.on_request_split()
    assert len(spike_ids) == 0

    a, b = 10, 100
    mouse_click(qtbot, v.canvas, (a, a), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (a, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, a), modifiers=('Control',))

    # Split lassoed points.
    spike_ids = v.on_request_split()
    assert len(spike_ids) > 0

    v.set_state(v.state)

    _stop_and_close(qtbot, v)
