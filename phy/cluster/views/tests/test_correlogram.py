"""Test correlogram view."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from phylib.io.mock import artificial_correlograms
from phylib.utils import connect, unconnect

from phy.plot.tests import mouse_click

from ..correlogram import CorrelogramView
from . import _stop_and_close

# ------------------------------------------------------------------------------
# Test correlogram view
# ------------------------------------------------------------------------------


def test_correlogram_view(qtbot, gui):
    def get_correlograms(cluster_ids, bin_size, window_size):
        return artificial_correlograms(len(cluster_ids), int(window_size / bin_size))

    def get_firing_rate(cluster_ids, bin_size):
        return 0.5 * np.ones((len(cluster_ids), len(cluster_ids)))

    v = CorrelogramView(
        correlograms=get_correlograms,
        firing_rate=get_firing_rate,
        sample_rate=100.0,
    )
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    deselected = []

    @connect(sender=v)
    def on_request_correlogram_deselect(sender, cluster_id_a, cluster_id_b):
        deselected.append((cluster_id_a, cluster_id_b))

    v.on_select(cluster_ids=[0, 2, 3])
    width, height = v.canvas.get_size()
    # A Control-secondary click on a diagonal autocorrelogram identifies one cluster.
    mouse_click(
        qtbot,
        v.canvas,
        (width / 2, height / 2),
        button='Right',
        modifiers=('Control',),
    )
    # Plain clicks do nothing; modified cross-correlogram clicks report both clusters.
    mouse_click(qtbot, v.canvas, (width / 6, height / 6), button='Right')
    mouse_click(
        qtbot,
        v.canvas,
        (width / 2, height / 6),
        button='Right',
        modifiers=('Control',),
    )

    assert deselected == [(2, 2), (0, 2)]
    unconnect(on_request_correlogram_deselect)

    v.toggle_normalization(True)
    v.toggle_labels(False)
    v.toggle_labels(True)

    v.set_bin(1)
    v.set_window(100)
    v.set_refractory_period(3)

    assert v.bin_size == 0.001
    assert v.window_size == 0.1
    assert v.refractory_period == 3e-3
    assert set(v.local_state_attrs) == {'bin_size', 'window_size', 'refractory_period'}

    v.increase()
    v.decrease()

    v.set_state(v.state)

    _stop_and_close(qtbot, v)
