# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import raises

from phylib.utils import Bunch
from phy.plot.tests import mouse_click
from ..scatter import ScatterView
from . import _stop_and_close


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

    v.increase_marker_size()
    v.decrease_marker_size()

    v.coords = lambda cluster_ids: 1
    with raises(ValueError):
        v.plot()

    _stop_and_close(qtbot, v)


def test_scatter_view_1(qtbot, gui):
    x = np.zeros(1)
    v = ScatterView(
        coords=lambda cluster_ids: Bunch(
            x=x, y=x, spike_ids=[0], spike_clusters=[0], data_bounds=(0, 0, 0, 0))
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.on_select(cluster_ids=[0])
    _stop_and_close(qtbot, v)


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

    _stop_and_close(qtbot, v)


def test_scatter_view_3(qtbot, gui):
    n = 1000
    v = ScatterView(
        coords=lambda cluster_ids: Bunch(
            pos=np.random.randn(n, 2),
            spike_ids=np.arange(n),
            spike_clusters=np.random.randint(size=n, low=0, high=4),
            data_bounds=None,
        )
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[0, 2])

    a, b = 50, 1000
    mouse_click(qtbot, v.canvas, (a, a), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (a, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, b), modifiers=('Control',))
    mouse_click(qtbot, v.canvas, (b, a), modifiers=('Control',))

    # Split lassoed points.
    spike_ids = v.on_request_split()
    assert len(spike_ids) > 0

    _stop_and_close(qtbot, v)
