# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import raises, yield_fixture

from phy.io.array import _spikes_per_cluster
from phy.io.mock import (artificial_waveforms,
                         artificial_features,
                         artificial_spike_clusters,
                         artificial_spike_samples,
                         artificial_masks,
                         artificial_traces,
                         )
from phy.gui import create_gui, GUIState
from phy.electrode.mea import staggered_positions
from phy.utils import Bunch
from ..views import TraceView, _extract_wave, _selected_clusters_colors


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _show(qtbot, view, stop=False):
    view.show()
    qtbot.waitForWindowShown(view.native)
    if stop:  # pragma: no cover
        qtbot.stop()
    view.close()


@yield_fixture(scope='session')
def model():
    model = Bunch()

    n_spikes = 51
    n_samples_w = 31
    n_samples_t = 20000
    n_channels = 11
    n_clusters = 3
    n_features = 4

    model.n_channels = n_channels
    model.n_spikes = n_spikes
    model.sample_rate = 20000.
    model.spike_times = artificial_spike_samples(n_spikes) * 1.
    model.spike_times /= model.spike_times[-1]
    model.spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    model.channel_positions = staggered_positions(n_channels)
    model.waveforms = artificial_waveforms(n_spikes, n_samples_w, n_channels)
    model.masks = artificial_masks(n_spikes, n_channels)
    model.traces = artificial_traces(n_samples_t, n_channels)
    model.features = artificial_features(n_spikes, n_channels, n_features)
    model.spikes_per_cluster = _spikes_per_cluster(model.spike_clusters)

    yield model


@contextmanager
def _test_view(view_name, model=None, tempdir=None):

    # Save a test GUI state JSON file in the tempdir.
    state = GUIState(config_dir=tempdir)
    state.set_view_params('WaveformView1', box_size=(.1, .1))
    state.set_view_params('TraceView1', box_size=(1., .01))
    state.set_view_params('FeatureView1', feature_scaling=.5)
    state.save()

    # Create the GUI.
    plugins = [view_name + 'Plugin']
    gui = create_gui(model=model, plugins=plugins, config_dir=tempdir)
    gui.show()

    v = gui.list_views(view_name)[0]

    # Select some spikes.
    spike_ids = np.arange(10)
    cluster_ids = np.unique(model.spike_clusters[spike_ids])
    v.on_select(cluster_ids, spike_ids)

    # Select other spikes.
    spike_ids = np.arange(2, 10)
    cluster_ids = np.unique(model.spike_clusters[spike_ids])
    v.on_select(cluster_ids, spike_ids)

    yield v

    gui.close()


#------------------------------------------------------------------------------
# Test utils
#------------------------------------------------------------------------------

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
        ac(v.boxed.box_size, (.1, .1), atol=1e-2)
        v.toggle_waveform_overlap()
        v.toggle_waveform_overlap()

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
        ac(v.stacked.box_size, (1., .01), atol=1e-2)

        v.go_to(.5)
        v.go_to(-.5)
        v.go_left()
        v.go_right()

        # qtbot.stop()


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot, model, tempdir):
    with _test_view('FeatureView', model=model, tempdir=tempdir) as v:

        assert v.feature_scaling == .5

        @v.set_best_channels_func
        def best_channels(cluster_id):
            return list(range(model.n_channels))

        v.add_attribute('sine', np.sin(np.linspace(-10., 10., model.n_spikes)))

        v.increase_feature_scaling()
        v.decrease_feature_scaling()

        # qtbot.stop()


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot, model, tempdir):
    with _test_view('CorrelogramView', model=model, tempdir=tempdir) as v:
        v.toggle_normalization()

    # qtbot.stop()
