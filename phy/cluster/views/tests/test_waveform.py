# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from numpy.testing import assert_allclose as ac
from vispy.util import keys

from .conftest import _select_clusters


#------------------------------------------------------------------------------
# Test waveform view
#------------------------------------------------------------------------------

def test_waveform_view(qtbot, gui):
    v = gui.controller.add_waveform_view(gui)
    _select_clusters(gui)

    v.toggle_waveform_overlap()
    v.toggle_waveform_overlap()

    v.toggle_zoom_on_channels()
    v.toggle_zoom_on_channels()

    v.toggle_show_labels()
    assert v.do_show_labels

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

    v.zoom_on_channels([0, 2, 4])

    v.filter_by_tag('test')

    # Simulate channel selection.
    _clicked = []

    @v.gui.connect_
    def on_channel_click(channel_idx=None, button=None, key=None):
        _clicked.append((channel_idx, button, key))

    v.events.key_press(key=keys.Key('2'))
    v.events.mouse_press(pos=(0., 0.), button=1)
    v.events.key_release(key=keys.Key('2'))

    assert _clicked == [(0, 1, 2)]

    # qtbot.stop()
    gui.close()
