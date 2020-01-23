# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.mock import artificial_waveforms
from phylib.utils import Bunch, connect

from phy.plot.tests import mouse_click
from ..cluscatter import ClusterScatterView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test cluster scatter view
#------------------------------------------------------------------------------

def test_cluster_scatter_view_1(qtbot, tempdir, gui):
    n_clusters = 1000
    cluster_ids = np.arange(n_clusters)

    def cluster_info(cluster_id):
        return Bunch({
            'fet1': np.random.randn(),
            'fet2': np.random.randn(),
            'fet3': np.random.uniform(low=5, high=20)
        })

    bindings = Bunch({'x_axis': 'fet1', 'y_axis': 'fet2', 'size': 'fet3'})

    v = ClusterScatterView(cluster_info=cluster_info, cluster_ids=cluster_ids, bindings=bindings)
    v.add_color_scheme(
        lambda cluster_id: np.random.rand(), name='depth',
        colormap='linear', cluster_ids=cluster_ids)
    v.color_scheme = 'depth'
    v.show()
    v.plot()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    _stop_and_close(qtbot, v)
