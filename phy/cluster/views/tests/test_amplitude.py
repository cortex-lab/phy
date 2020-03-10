# -*- coding: utf-8 -*-

"""Test amplitude view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.mock import artificial_spike_samples
from phylib.utils import Bunch, connect

from phy.plot.tests import mouse_click
from ..amplitude import AmplitudeView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test amplitude view
#------------------------------------------------------------------------------

def test_amplitude_view_0(qtbot, gui):
    v = AmplitudeView(
        amplitudes=lambda cluster_ids, load_all=False: None,
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.on_select(cluster_ids=[0])

    v.increase_marker_size()
    v.decrease_marker_size()
    v.reset_marker_size()

    _stop_and_close(qtbot, v)


def test_amplitude_view_1(qtbot, gui):
    x = np.zeros(1)
    v = AmplitudeView(
        amplitudes=lambda cluster_ids, load_all=False: [
            Bunch(amplitudes=x, spike_ids=[0], spike_times=[0])],
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)
    v.on_select(cluster_ids=[0])

    v.show_time_range((.499, .501))

    _stop_and_close(qtbot, v)


def test_amplitude_view_2(qtbot, gui):
    n = 1000
    st1 = artificial_spike_samples(n) / 20000.
    st2 = artificial_spike_samples(n) / 20000.
    v = AmplitudeView(
        amplitudes={
            'amp1': lambda cluster_ids, load_all=False: [Bunch(
                amplitudes=15 + np.random.randn(n),
                spike_ids=np.arange(n),
                spike_times=st1,
            ) for c in cluster_ids],
            'amp2': lambda cluster_ids, load_all=False: [Bunch(
                amplitudes=10 + np.random.randn(n),
                spike_ids=np.arange(n),
                spike_times=st2,
            ) for c in cluster_ids],
        }, duration=max(st1.max(), st2.max()))
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.next_amplitudes_type()
    v.previous_amplitudes_type()
    v.actions.change_amplitudes_type_to_amp2()

    v.set_state(v.state)

    w, h = v.canvas.get_size()

    _times = []

    @connect(sender=v)
    def on_select_time(sender, time):
        _times.append(time)
    mouse_click(qtbot, v.canvas, (w / 3, h / 2), modifiers=('Alt',))
    assert len(_times) == 1
    assert np.allclose(_times[0], .5, atol=.01)

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
