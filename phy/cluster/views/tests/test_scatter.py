# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import Bunch
from phy.plot.tests import mouse_click
from ..scatter import ScatterView


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

def test_scatter_view_0(qtbot, gui):
    v = ScatterView(
        coords=lambda cluster_ids, load_all=False: None
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.on_select(cluster_ids=[0])

    v.increase()
    v.decrease()

    v.close()


def test_scatter_view_1(qtbot, gui):
    x = np.zeros(1)
    v = ScatterView(
        coords=lambda cluster_ids, load_all=False: [
            Bunch(x=x, y=x, spike_ids=[0], data_bounds=(0, 0, 0, 0))]
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.on_select(cluster_ids=[0])
    v.close()


def test_scatter_view_2(qtbot, gui):
    n = 1000
    v = ScatterView(
        coords=lambda cluster_ids, load_all=False: [Bunch(
            x=np.random.randn(n),
            y=np.random.randn(n),
            spike_ids=np.arange(n),
            data_bounds=None,
        ) for c in cluster_ids]
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.set_state(v.state)

    # Split without selection.
    spike_ids = v.on_request_split()
    assert len(spike_ids) == 0

    a, b = 50, 1000
    mouse_click(qtbot, v.canvas, (a, a), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (a, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, a), modifiers=('Control',))

    # Split lassoed points.
    spike_ids = v.on_request_split()
    assert len(spike_ids) > 0

    # qtbot.stop()
    v.close()
