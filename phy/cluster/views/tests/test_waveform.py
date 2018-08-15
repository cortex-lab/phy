# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_allclose as ac
from vispy.util import keys

from phy.electrode.mea import staggered_positions
from phy.gui import GUI
from phy.io.mock import artificial_waveforms
from phy.utils import Bunch, connect

from ..waveform import WaveformView


#------------------------------------------------------------------------------
# Test waveform view
#------------------------------------------------------------------------------

def test_waveform_view(qtbot, tempdir):
    nc = 5

    def get_waveforms(cluster_id):
        return Bunch(data=artificial_waveforms(10, 20, nc),
                     channel_ids=np.arange(nc),
                     channel_positions=staggered_positions(nc),
                     )

    v = WaveformView(waveforms=get_waveforms,
                     )
    gui = GUI(config_dir=tempdir)
    v.attach(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.toggle_waveform_overlap(True)
    v.toggle_waveform_overlap(False)

    v.toggle_show_labels(True)
    v.toggle_show_labels(False)

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

    v.events.key_press(key=keys.Key('2'))
    v.events.mouse_press(pos=(0., 0.), button=1)
    v.events.key_release(key=keys.Key('2'))

    assert _clicked == [(0, 1, 2)]

    # qtbot.stop()
    gui.close()
