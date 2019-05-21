# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import Bunch
from phylib.utils._color import ClusterColorSelector
from phylib.io.mock import artificial_spike_clusters, artificial_spike_samples

from ..raster import RasterView


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

    # qtbot.stop()
    v.close()


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

    c.set_color_mapping('group', 'cluster_group')
    v.plot()

    # qtbot.stop()
    v.close()
