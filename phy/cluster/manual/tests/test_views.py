# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac
from vispy.util import keys
from pytest import fixture

from phy.utils import Bunch
from .conftest import MockController
from ..views import (ScatterView,
                     _extract_wave,
                     _extend,
                     )


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

@fixture
def state(tempdir):
    # Save a test GUI state JSON file in the tempdir.
    state = Bunch()
    state.WaveformView0 = Bunch(overlap=False)
    state.TraceView0 = Bunch(scaling=1.)
    state.FeatureView0 = Bunch(feature_scaling=.5)
    state.CorrelogramView0 = Bunch(uniform_normalization=True)
    return state


@fixture
def gui(tempdir, state):
    controller = MockController(config_dir=tempdir)
    return controller.create_gui(add_default_views=False, **state)


def _select_clusters(gui):
    gui.show()
    mc = gui.controller.manual_clustering
    assert mc
    mc.select([])
    mc.select([0])
    mc.select([0, 2])
    mc.select([0, 2, 3])


#------------------------------------------------------------------------------
# Test utils
#------------------------------------------------------------------------------

def test_extend():
    l = list(range(5))
    assert _extend(l) == l
    assert _extend(l, 0) == []
    assert _extend(l, 4) == list(range(4))
    assert _extend(l, 5) == l
    assert _extend(l, 6) == (l + [4])


def test_extract_wave():
    traces = np.arange(30).reshape((6, 5))
    mask = np.array([0, 1, 1, .5, 0])
    wave_len = 4
    hwl = wave_len // 2

    ae(_extract_wave(traces, 0 - hwl, mask, wave_len)[0],
       [[0, 0], [0, 0], [1, 2], [6, 7]])

    ae(_extract_wave(traces, 1 - hwl, mask, wave_len)[0],
       [[0, 0], [1, 2], [6, 7], [11, 12]])

    ae(_extract_wave(traces, 2 - hwl, mask, wave_len)[0],
       [[1, 2], [6, 7], [11, 12], [16, 17]])

    ae(_extract_wave(traces, 5 - hwl, mask, wave_len)[0],
       [[16, 17], [21, 22], [0, 0], [0, 0]])


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

    # Simulate channel selection.
    _clicked = []

    @v.gui.connect_
    def on_channel_click(channel_idx=None, button=None, key=None):
        _clicked.append((channel_idx, button, key))

    v.events.key_press(key=keys.Key('2'))
    v.events.mouse_press(pos=(0., 0.), button=1)
    v.events.key_release(key=keys.Key('2'))

    assert _clicked == [(0, 1, 2)]

    v.next_data()

    # qtbot.stop()
    gui.close()


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
    v.widen()
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


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot, gui):
    v = gui.controller.add_feature_view(gui)
    _select_clusters(gui)

    assert v.feature_scaling == .5
    v.add_attribute('sine',
                    np.sin(np.linspace(-10., 10., gui.controller.n_spikes)))

    v.increase()
    v.decrease()

    v.on_channel_click(channel_idx=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # qtbot.stop()
    gui.close()


#------------------------------------------------------------------------------
# Test scatter view
#------------------------------------------------------------------------------

def test_scatter_view(qtbot, gui):
    n = 1000
    v = ScatterView(coords=lambda c: Bunch(x=np.random.randn(n),
                                           y=np.random.randn(n),
                                           spike_ids=np.arange(n),
                                           spike_clusters=np.ones(n).
                                           astype(np.int32) * c[0],
                                           ) if 2 not in c else None,
                    # data_bounds=[-3, -3, 3, 3],
                    )
    v.attach(gui)

    _select_clusters(gui)

    # qtbot.stop()
    gui.close()


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot, gui):
    v = gui.controller.add_correlogram_view(gui)
    _select_clusters(gui)

    v.toggle_normalization()

    v.set_bin(1)
    v.set_window(100)

    # qtbot.stop()
    gui.close()
