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

from phy.electrode.mea import staggered_positions
from phy.gui import create_gui
from phy.io.array import _spikes_per_cluster
from phy.io.mock import (artificial_waveforms,
                         artificial_features,
                         artificial_masks,
                         artificial_traces,
                         )
from phy.io.store import ClusterStore, get_closest_clusters
from phy.stats.clusters import (mean,
                                get_max_waveform_amplitude,
                                get_mean_masked_features_distance,
                                get_unmasked_channels,
                                get_sorted_main_channels,
                                )
from phy.utils import Bunch
from ..gui_component import ManualClustering
from ..views import (WaveformView,
                     FeatureView,
                     CorrelogramView,
                     TraceView,
                     _extract_wave,
                     _selected_clusters_colors,
                     _extend,
                     )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def create_model():
    model = Bunch()

    n_samples_waveforms = 31
    n_samples_t = 20000
    n_channels = 11
    n_clusters = 3
    model.n_spikes_per_cluster = 51
    n_spikes_total = n_clusters * model.n_spikes_per_cluster
    n_features_per_channel = 4

    model.n_channels = n_channels
    model.n_spikes = n_spikes_total
    model.sample_rate = 20000.
    model.duration = n_samples_t / float(model.sample_rate)
    model.spike_times = np.linspace(0., model.duration, n_spikes_total)
    model.spike_clusters = np.repeat(np.arange(n_clusters),
                                     model.n_spikes_per_cluster)
    model.cluster_ids = np.unique(model.spike_clusters)
    model.channel_positions = staggered_positions(n_channels)
    model.traces = artificial_traces(n_samples_t, n_channels)
    model.masks = artificial_masks(n_spikes_total, n_channels)

    model.spikes_per_cluster = _spikes_per_cluster(model.spike_clusters)
    model.n_features_per_channel = n_features_per_channel
    model.n_samples_waveforms = n_samples_waveforms
    model.cluster_groups = {c: None for c in range(n_clusters)}

    # TODO: make this cleaner by abstracting the store away
    model.store = create_cluster_store(model)

    return model


def create_cluster_store(model):
    cs = ClusterStore()

    def get_waveforms(n):
        return artificial_waveforms(n,
                                    model.n_samples_waveforms,
                                    model.n_channels)

    def get_masks(n):
        return artificial_masks(n, model.n_channels)

    def get_features(n):
        return artificial_features(n,
                                   model.n_channels,
                                   model.n_features_per_channel)

    def get_spike_ids(cluster_id):
        n = model.n_spikes_per_cluster
        return np.arange(n) + n * cluster_id

    def _get_data(**kwargs):
        kwargs['spike_clusters'] = model.spike_clusters[kwargs['spike_ids']]
        return Bunch(**kwargs)

    @cs.add(concat=True)
    def masks(cluster_id):
        return _get_data(spike_ids=get_spike_ids(cluster_id),
                         masks=get_masks(model.n_spikes_per_cluster))

    @cs.add(concat=True)
    def features(cluster_id):
        return _get_data(spike_ids=get_spike_ids(cluster_id),
                         features=get_features(model.n_spikes_per_cluster))

    @cs.add(concat=True)
    def features_masks(cluster_id):
        return _get_data(spike_ids=get_spike_ids(cluster_id),
                         features=get_features(model.n_spikes_per_cluster),
                         masks=get_masks(model.n_spikes_per_cluster))

    @cs.add
    def feature_lim():
        """Return the max of a subset of the feature amplitudes."""
        return 1

    @cs.add
    def background_features_masks():
        f = get_features(model.n_spikes)
        m = model.masks
        return _get_data(spike_ids=np.arange(model.n_spikes),
                         features=f, masks=m)

    @cs.add(concat=True)
    def waveforms(cluster_id):
        return _get_data(spike_ids=get_spike_ids(cluster_id),
                         waveforms=get_waveforms(model.n_spikes_per_cluster))

    @cs.add
    def waveform_lim():
        """Return the max of a subset of the waveform amplitudes."""
        return 1

    @cs.add(concat=True)
    def waveforms_masks(cluster_id):
        w = get_waveforms(model.n_spikes_per_cluster)
        m = get_masks(model.n_spikes_per_cluster)
        mw = cs.mean_waveforms(cluster_id)[np.newaxis, ...]
        mm = cs.mean_masks(cluster_id)[np.newaxis, ...]
        return _get_data(spike_ids=get_spike_ids(cluster_id),
                         waveforms=w,
                         masks=m,
                         mean_waveforms=mw,
                         mean_masks=mm,
                         )

    # Mean quantities.
    # -------------------------------------------------------------------------

    @cs.add
    def mean_masks(cluster_id):
        # We access [1] because we return spike_ids, masks.
        return mean(cs.masks(cluster_id).masks)

    @cs.add
    def mean_features(cluster_id):
        return mean(cs.features(cluster_id).features)

    @cs.add
    def mean_waveforms(cluster_id):
        return mean(cs.waveforms(cluster_id).waveforms)

    # Statistics.
    # -------------------------------------------------------------------------

    @cs.add(cache='memory')
    def best_channels(cluster_id):
        mm = cs.mean_masks(cluster_id)
        uch = get_unmasked_channels(mm)
        return get_sorted_main_channels(mm, uch)

    @cs.add(cache='memory')
    def best_channels_multiple(cluster_ids):
        best_channels = []
        for cluster in cluster_ids:
            channels = cs.best_channels(cluster)
            best_channels.extend([ch for ch in channels
                                  if ch not in best_channels])
        return best_channels

    @cs.add(cache='memory')
    def max_waveform_amplitude(cluster_id):
        mm = cs.mean_masks(cluster_id)
        mw = cs.mean_waveforms(cluster_id)
        assert mw.ndim == 2
        return np.asscalar(get_max_waveform_amplitude(mm, mw))

    @cs.add(cache=None)
    def mean_masked_features_score(cluster_0, cluster_1):
        mf0 = cs.mean_features(cluster_0)
        mf1 = cs.mean_features(cluster_1)
        mm0 = cs.mean_masks(cluster_0)
        mm1 = cs.mean_masks(cluster_1)
        nfpc = model.n_features_per_channel
        d = get_mean_masked_features_distance(mf0, mf1, mm0, mm1,
                                              n_features_per_channel=nfpc)
        s = 1. / max(1e-10, d)
        return s

    @cs.add(cache='memory')
    def most_similar_clusters(cluster_id):
        assert isinstance(cluster_id, int)
        return get_closest_clusters(cluster_id, model.cluster_ids,
                                    cs.mean_masked_features_score)

    # Traces.
    # -------------------------------------------------------------------------

    @cs.add
    def mean_traces():
        mt = model.traces[:, :].mean(axis=0)
        return mt.astype(model.traces.dtype)

    return cs


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _show(qtbot, view, stop=False):
    view.show()
    qtbot.waitForWindowShown(view.native)
    if stop:  # pragma: no cover
        qtbot.stop()
    view.close()


@fixture
def state(tempdir):
    # Save a test GUI state JSON file in the tempdir.
    state = Bunch()
    state.WaveformView0 = Bunch(overlap=False)
    state.TraceView0 = Bunch(scaling=1.)
    state.FeatureView0 = Bunch(feature_scaling=.5)
    state.CorrelogramView0 = Bunch(uniform_normalization=True)

    # quality and similarity functions for the cluster view.
    state.ClusterView = Bunch(quality='max_waveform_amplitude',
                              similarity='most_similar_clusters')
    return state


@fixture
def gui(tempdir, state):
    model = create_model()
    gui = create_gui(model=model, config_dir=tempdir, **state)
    mc = ManualClustering(model.spike_clusters,
                          cluster_groups=model.cluster_groups,)
    mc.attach(gui)
    gui.register(manual_clustering=mc)
    return gui


def _select_clusters(gui):
    gui.show()
    mc = gui.request('manual_clustering')
    assert mc
    mc.select([])
    mc.select([0])
    mc.select([0, 2])


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

    ae(_extract_wave(traces, 0 - hwl, 0 + hwl, mask, wave_len)[0],
       [[0, 0, 0], [0, 0, 0], [1, 2, 3], [6, 7, 8]])

    ae(_extract_wave(traces, 1 - hwl, 1 + hwl, mask, wave_len)[0],
       [[0, 0, 0], [1, 2, 3], [6, 7, 8], [11, 12, 13]])

    ae(_extract_wave(traces, 2 - hwl, 2 + hwl, mask, wave_len)[0],
       [[1, 2, 3], [6, 7, 8], [11, 12, 13], [16, 17, 18]])

    ae(_extract_wave(traces, 5 - hwl, 5 + hwl, mask, wave_len)[0],
       [[16, 17, 18], [21, 22, 23], [0, 0, 0], [0, 0, 0]])


def test_selected_clusters_colors():
    assert _selected_clusters_colors().shape[0] > 10
    assert _selected_clusters_colors(0).shape[0] == 0
    assert _selected_clusters_colors(1).shape[0] == 1
    assert _selected_clusters_colors(100).shape[0] == 100


#------------------------------------------------------------------------------
# Test waveform view
#------------------------------------------------------------------------------

def test_waveform_view(qtbot, gui):
    v = WaveformView(waveforms_masks=gui.model.store.waveforms_masks,
                     channel_positions=gui.model.channel_positions,
                     n_samples=gui.model.n_samples_waveforms,
                     waveform_lim=gui.model.store.waveform_lim(),
                     best_channels=gui.model.store.best_channels_multiple,
                     )
    v.attach(gui)

    _select_clusters(gui)

    ac(v.boxed.box_size, (.1818, .0909), atol=1e-2)

    v.toggle_waveform_overlap()
    v.toggle_waveform_overlap()

    v.toggle_show_means()
    v.toggle_show_means()

    v.toggle_zoom_on_channels()
    v.toggle_zoom_on_channels()

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

    # qtbot.stop()
    gui.close()


#------------------------------------------------------------------------------
# Test trace view
#------------------------------------------------------------------------------

def test_trace_view_no_spikes(qtbot):
    n_samples = 1000
    n_channels = 12
    sample_rate = 2000.

    traces = artificial_traces(n_samples, n_channels)
    mt = np.atleast_2d(traces.mean(axis=0))

    # Create the view.
    v = TraceView(traces=traces, sample_rate=sample_rate, mean_traces=mt)
    _show(qtbot, v)


def test_trace_view_spikes(qtbot, gui):
    model = gui.model
    cs = model.store
    v = TraceView(traces=model.traces,
                  sample_rate=model.sample_rate,
                  spike_times=model.spike_times,
                  spike_clusters=model.spike_clusters,
                  n_samples_per_spike=model.n_samples_waveforms,
                  masks=model.masks,
                  mean_traces=cs.mean_traces(),
                  )
    v.attach(gui)

    _select_clusters(gui)

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

    v.origin = 'upper'
    assert v.origin == 'upper'

    # qtbot.stop()
    gui.close()


#------------------------------------------------------------------------------
# Test feature view
#------------------------------------------------------------------------------

def test_feature_view(qtbot, gui):
    model = gui.model
    cs = model.store
    v = FeatureView(features_masks=cs.features_masks,
                    background_features_masks=cs.background_features_masks(),
                    spike_times=model.spike_times,
                    n_channels=model.n_channels,
                    n_features_per_channel=model.n_features_per_channel,
                    feature_lim=cs.feature_lim(),
                    )
    v.attach(gui)

    _select_clusters(gui)

    assert v.feature_scaling == .5
    v.add_attribute('sine',
                    np.sin(np.linspace(-10., 10., model.n_spikes)))

    v.increase()
    v.decrease()

    v.on_channel_click(channel_idx=3, button=1, key=2)
    v.clear_channels()
    v.toggle_automatic_channel_selection()

    # qtbot.stop()
    gui.close()


#------------------------------------------------------------------------------
# Test correlogram view
#------------------------------------------------------------------------------

def test_correlogram_view(qtbot, gui):
    model = gui.model
    v = CorrelogramView(spike_times=model.spike_times,
                        spike_clusters=model.spike_clusters,
                        sample_rate=model.sample_rate,
                        )
    v.attach(gui)
    _select_clusters(gui)

    v.toggle_normalization()

    v.set_bin(1)
    v.set_window(100)

    # qtbot.stop()
    gui.close()
