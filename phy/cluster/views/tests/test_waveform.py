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


#------------------------------------------------------------------------------
# Test waveform view
#------------------------------------------------------------------------------

def test_waveform_view(qtbot, tempdir, gui):
    nc = 5

    w = artificial_waveforms(10, 20, nc)

    def get_waveforms(cluster_id):
        return Bunch(data=w,
                     channel_ids=np.arange(nc),
                     channel_positions=staggered_positions(nc),
                     )

    v = WaveformView(waveforms=get_waveforms,
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

    # Box scaling.
    bs = v.boxed.box_size
    v.increase()
    v.decrease()
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
    def on_channel_click(sender, channel_id=None, button=None, key=None):
        _clicked.append((channel_id, button, key))

    key_press(qtbot, v.canvas, '2')
    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Left')
    key_release(qtbot, v.canvas, '2')

    assert _clicked == [(2, 'Left', 2)]

    v.set_state(v.state)

    # qtbot.stop()
    v.close()
