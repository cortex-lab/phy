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
    spike_times = np.arange(4)
    spike_clusters = np.arange(4)
    cluster_ids = np.arange(4)

    cluster_meta = Bunch(fields=('group',), get=lambda f, cl: cl % 4)
    cluster_metrics = {'quality': lambda c: 3. - c}
    c = ClusterColorSelector(
        cluster_meta=cluster_meta,
        cluster_metrics=cluster_metrics,
        cluster_ids=cluster_ids)

    v = RasterView(spike_times, spike_clusters, cluster_color_selector=c)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.plot()

    v.on_select([2])
    c.set_color_mapping('quality', 'categorical')
    v.plot()

    v.increase()
    v.decrease()

    # Simulate channel selection.
    _clicked = []

    @connect(sender=v)
    def on_cluster_click(sender, cluster_id=None, button=None):
        _clicked.append((cluster_id, button))

    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Left', modifiers=('Control',))

    assert _clicked == [(1, 'Left')]

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

    v = RasterView(spike_times, spike_clusters, cluster_color_selector=c)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.plot()
    v.on_select([0])

    v.update_cluster_sort(np.arange(nc))

    c.set_color_mapping('group', 'cluster_group')
    v.update_color()

    v.set_cluster_ids(np.arange(0, nc, 2))
    v.plot()

    _stop_and_close(qtbot, v)
