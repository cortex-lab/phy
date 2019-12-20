# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import Bunch, connect
from phylib.utils.color import ClusterColorSelector
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

    cluster_meta = Bunch(fields=('group',), get=lambda f, cl: cl % 4)
    cluster_metrics = {'quality': lambda c: 3. - c}
    c = ClusterColorSelector(
        cluster_meta=cluster_meta,
        cluster_metrics=cluster_metrics,
        cluster_ids=cluster_ids)

    class Supervisor(object):
        pass
    s = Supervisor()

    v = RasterView(spike_times, spike_clusters, cluster_color_selector=c)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.set_cluster_ids(cluster_ids)
    v.plot()

    v.on_select(cluster_ids=[2], sender=s)
    c.set_color_mapping('quality', 'categorical')
    v.plot()

    v.increase_marker_size()
    v.decrease_marker_size()

    # Simulate cluster selection.
    _clicked = []

    w, h = v.canvas.get_size()

    @connect(sender=v)
    def on_select(sender, cluster_ids):
        _clicked.append(cluster_ids)

    @connect(sender=v)
    def on_select_more(sender, cluster_ids):
        _clicked.append(cluster_ids)

    mouse_click(qtbot, v.canvas, pos=(w / 2, 0.), button='Left', modifiers=('Control',))
    assert len(_clicked) == 1
    assert _clicked == [[0]]

    mouse_click(qtbot, v.canvas, pos=(w / 2, h / 2), button='Left', modifiers=('Shift',))
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

    cluster_meta = Bunch(fields=('group',), get=lambda f, cl: cl % 4)
    cluster_metrics = {'quality': lambda c: 100 - c}
    c = ClusterColorSelector(
        cluster_meta=cluster_meta,
        cluster_metrics=cluster_metrics,
        cluster_ids=cluster_ids)

    class Supervisor(object):
        pass
    s = Supervisor()

    v = RasterView(spike_times, spike_clusters, cluster_color_selector=c)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.set_cluster_ids(cluster_ids)
    v.plot()
    v.on_select(cluster_ids=[0], sender=s)

    v.update_cluster_sort(np.arange(nc))

    c.set_color_mapping('group', 'cluster_group')
    v.update_color()

    v.set_cluster_ids(np.arange(0, nc, 2))
    v.plot()

    _stop_and_close(qtbot, v)
