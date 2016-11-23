# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from numpy.testing import assert_allclose as ac

from .conftest import _select_clusters


#------------------------------------------------------------------------------
# Test trace view
#------------------------------------------------------------------------------

def test_trace_view(qtbot, gui):
    v = gui.controller.add_trace_view(gui)

    _select_clusters(gui)

    ac(v.stacked.box_size, (1., .08181), atol=1e-3)
    assert v.time == .5

    v.go_to(.25)
    assert v.time == .25

    v.go_to(-.5)
    assert v.time == .125

    v.go_left()
    assert v.time == .125

    v.go_right()
    assert v.time == .175

    # Change interval size.
    v.interval = (.25, .75)
    ac(v.interval, (.25, .75))
    v.widen()
    ac(v.interval, (.125, .875))
    v.narrow()
    ac(v.interval, (.25, .75))

    # Widen the max interval.
    v.set_interval((0, gui.controller.duration))
    v.widen()

    v.toggle_show_labels()
    v.go_right()
    assert v.do_show_labels

    # Change channel scaling.
    bs = v.stacked.box_size
    v.increase()
    v.decrease()
    ac(v.stacked.box_size, bs, atol=1e-3)

    v.origin = 'upper'
    assert v.origin == 'upper'

    # qtbot.stop()
    gui.close()
