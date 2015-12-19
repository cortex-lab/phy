# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import raises
from vispy.util import keys

from phy.gui import create_gui, GUIState
from phy.io.mock import artificial_traces
from ..views import (TraceView, _extract_wave, _selected_clusters_colors,
                     _extend)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _show(qtbot, view, stop=False):
    view.show()
    qtbot.waitForWindowShown(view.native)
    if stop:  # pragma: no cover
        qtbot.stop()
    view.close()


@contextmanager
def _test_view(view_name, model=None, tempdir=None):

    # Save a test GUI state JSON file in the tempdir.
    state = GUIState(config_dir=tempdir)
    state.set_view_params('WaveformView1', overlap=False, box_size=(.1, .1))
    state.set_view_params('TraceView1', box_size=(1., .01))
    state.set_view_params('FeatureView1', feature_scaling=.5)
    state.set_view_params('CorrelogramView1', uniform_normalization=True)
    state.save()

    # Create the GUI.
    plugins = ['ManualClusteringPlugin',
               view_name + 'Plugin']
    gui = create_gui(model=model, plugins=plugins, config_dir=tempdir)
    gui.show()

    gui.manual_clustering.select([])
    gui.manual_clustering.select([0])
    gui.manual_clustering.select([0, 2])

    view = gui.list_views(view_name)[0]
    view.gui = gui
    yield view

    gui.close()


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

    with raises(ValueError):
        _extract_wave(traces, -1, mask, wave_len)

    with raises(ValueError):
        _extract_wave(traces, 20, mask, wave_len)

    ae(_extract_wave(traces, 0, mask, wave_len)[0],
       [[0, 0, 0], [0, 0, 0], [1, 2, 3], [6, 7, 8]])

    ae(_extract_wave(traces, 1, mask, wave_len)[0],
       [[0, 0, 0], [1, 2, 3], [6, 7, 8], [11, 12, 13]])

    ae(_extract_wave(traces, 2, mask, wave_len)[0],
       [[1, 2, 3], [6, 7, 8], [11, 12, 13], [16, 17, 18]])

    ae(_extract_wave(traces, 5, mask, wave_len)[0],
       [[16, 17, 18], [21, 22, 23], [0, 0, 0], [0, 0, 0]])


def test_selected_clusters_colors():
    assert _selected_clusters_colors().shape[0] > 10
    assert _selected_clusters_colors(0).shape[0] == 0
    assert _selected_clusters_colors(1).shape[0] == 1
    assert _selected_clusters_colors(100).shape[0] == 100


#------------------------------------------------------------------------------
# Test waveform view
#------------------------------------------------------------------------------

def test_waveform_view(qtbot, model, tempdir):
    with _test_view('WaveformView', model=model, tempdir=tempdir) as v:
        ac(v.boxed.box_size, (.1818, .0909), atol=1e-2)
        v.toggle_waveform_overlap()
        v.toggle_waveform_overlap()

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

        # qtbot.stop()


#------------------------------------------------------------------------------
# Test trace view
#------------------------------------------------------------------------------

def test_trace_view_no_spikes(qtbot):
    n_samples = 1000
    n_channels = 12
    sample_rate = 2000.

    traces = artificial_traces(n_samples, n_channels)

    # Create the view.
    v = TraceView(traces=traces, sample_rate=sample_rate)
    _show(qtbot, v)


def test_trace_view_spikes(qtbot, model, tempdir):
    with _test_view('TraceView', model=model, tempdir=tempdir) as v:
        ac(v.stacked.box_size, (1., .08181), atol=1e-3)
        assert v.time == .25

        v.go_to(.5)
        assert v.time == .5

        v.go_to(-.5)
        assert v.time == .25

        v.go_left()
        assert v.time == .25

        v.go_right()
        assert v.time == .35

        # Change interval size.
        v.set_interval((.25, .75))
        ac(v.interval, (.25, .75))
        v.widen()
        ac(v.interval, (.225, .775))
        v.narrow()
        ac(v.interval, (.25, .75))

        # Widen the max interval.
        v.set_interval((0, model.duration))
        v.widen()

        # Change channel scaling.
        bs = v.stacked.box_size
        v.increase()
        v.decrease()
        ac(v.stacked.box_size, bs, atol=1e-3)

        # qtbot.stop()


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot, model, tempdir):
    with _test_view('FeatureView', model=model, tempdir=tempdir) as v:
        assert v.feature_scaling == .5
        v.add_attribute('sine', np.sin(np.linspace(-10., 10., model.n_spikes)))

        v.increase()
        v.decrease()

        v.on_channel_click(channel_idx=3, button=1, key=2)
        v.clear_channels()

        # qtbot.stop()


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot, model, tempdir):
    with _test_view('CorrelogramView', model=model, tempdir=tempdir) as v:
        v.toggle_normalization()

        v.set_bin(1)
        v.set_window(100)
        # qtbot.stop()
