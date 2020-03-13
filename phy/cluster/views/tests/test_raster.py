# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import connect
from phylib.io.mock import artificial_spike_clusters, artificial_spike_samples

from phy.plot.tests import mouse_click
from ..raster import RasterView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

def test_raster_0(qtbot, gui):
    n = 5
    spike_times = np.arange(n)
    spike_clusters = np.arange(n)
    cluster_ids = np.arange(n)

    class Supervisor(object):
        pass
    s = Supervisor()

    v = RasterView(spike_times, spike_clusters)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.set_cluster_ids(cluster_ids)
    v.plot()
    assert v.status

    v.on_select(cluster_ids=[2], sender=s)
    v.plot()

    v.increase_marker_size()
    v.decrease_marker_size()

    # Simulate cluster selection.
    _clicked = []

    w, h = v.canvas.get_size()

    @connect(sender=v)
    def on_request_select(sender, cluster_ids):
        _clicked.append(cluster_ids)

    @connect(sender=v)
    def on_select_more(sender, cluster_ids):
        _clicked.append(cluster_ids)

    mouse_click(qtbot, v.canvas, pos=(w / 2, 0.), button='Left', modifiers=('Control',))
    assert len(_clicked) == 1
    assert _clicked == [[0]]

    mouse_click(
        qtbot, v.canvas, pos=(w / 2, h / 2), button='Left', modifiers=('Control', 'Shift',))
    assert len(_clicked) == 2
    assert _clicked[1][0] in (1, 2)

    v.zoom_to_time_range((1., 3.))

    _stop_and_close(qtbot, v)


def test_raster_1(qtbot, gui):
    ns = 10000
    nc = 100
    spike_times = artificial_spike_samples(ns) / 20000.
    spike_clusters = artificial_spike_clusters(ns, nc)
    cluster_ids = np.arange(4)

    v = RasterView(spike_times, spike_clusters)

    @v.add_color_scheme(
        name='group', cluster_ids=cluster_ids,
        colormap='cluster_group', categorical=True)
    def cg(cluster_id):
        return cluster_id % 4

    v.add_color_scheme(
        lambda cid: cid, name='random', cluster_ids=cluster_ids,
        colormap='categorical', categorical=True)

    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.set_cluster_ids(cluster_ids)
    v.plot()
    v.on_select(cluster_ids=[0])

    v.update_cluster_sort(np.arange(nc))

    v.set_cluster_ids(np.arange(0, nc, 2))
    v.update_color()
    v.plot()

    _stop_and_close(qtbot, v)
