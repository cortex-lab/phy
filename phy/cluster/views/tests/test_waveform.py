# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_allclose as ac

from phylib.io.mock import artificial_waveforms
from phylib.utils import Bunch, connect
from phylib.utils.geometry import staggered_positions
from phy.plot.tests import mouse_click, key_press, key_release

from ..waveform import WaveformView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test waveform view
#------------------------------------------------------------------------------

def test_waveform_view(qtbot, tempdir, gui):
    nc = 5
    ns = 10

    w = 10 + 100 * artificial_waveforms(ns, 20, nc)

    def get_waveforms(cluster_id):
        return Bunch(
            data=w,
            masks=np.random.uniform(low=0., high=1., size=(ns, nc)),
            channel_ids=np.arange(nc),
            channel_labels=['%d' % (ch * 10) for ch in range(nc)],
            channel_positions=staggered_positions(nc))

    v = WaveformView(
        waveforms={'waveforms': get_waveforms, 'mean_waveforms': get_waveforms},
        sample_rate=10000.,
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.toggle_waveform_overlap(True)
    v.toggle_waveform_overlap(False)

    v.toggle_show_labels(False)
    v.toggle_show_labels(True)

    v.next_waveforms_type()
    v.previous_waveforms_type()
    v.toggle_mean_waveforms(True)
    v.toggle_mean_waveforms(False)

    # Box scaling.
    bs = v.boxed.box_size
    v.increase()
    v.decrease()
    v.reset_scaling()
    ac(v.boxed.box_size, bs)

    bs = v.boxed.box_size
    v.widen()
    v.narrow()
    ac(v.boxed.box_size, bs)

    # Probe scaling.
    bp = v.boxed.box_pos
    v.extend_horizontally()
    v.shrink_horizontally()
    ac(v.boxed.box_pos, bp)

    bp = v.boxed.box_pos
    v.extend_vertically()
    v.shrink_vertically()
    ac(v.boxed.box_pos, bp)

    a, b = v.probe_scaling
    v.probe_scaling = (a, b * 2)
    ac(v.probe_scaling, (a, b * 2))

    a, b = v.box_scaling
    v.box_scaling = (a * 2, b)
    ac(v.box_scaling, (a * 2, b))

    # Simulate channel selection.
    _clicked = []

    @connect(sender=v)
    def on_select_channel(sender, channel_id=None, button=None, key=None):
        _clicked.append((channel_id, button, key))

    key_press(qtbot, v.canvas, '2')
    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Left')
    key_release(qtbot, v.canvas, '2')

    assert _clicked == [(2, 'Left', 2)]

    v.set_state(v.state)

    _stop_and_close(qtbot, v)
