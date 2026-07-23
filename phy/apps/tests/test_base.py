"""Integration tests for the GUIs."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import logging
import os
import shutil
import tempfile
import unittest
from itertools import cycle, islice
from pathlib import Path

import numpy as np
from phylib.io.array import SpikeSelector, _sample_spikes_evenly
from phylib.io.mock import (
    artificial_features,
    artificial_spike_clusters,
    artificial_spike_samples,
    artificial_traces,
    artificial_waveforms,
)
from phylib.utils import Bunch, connect, emit, reset, unconnect
from pytest import mark
from pytestqt.plugin import QtBot

from phy.cluster.clustering import Clustering
from phy.cluster.views import (
    AmplitudeView,
    FeatureView,
    TemplateView,
    TraceView,
    WaveformView,
)
from phy.gui.qt import Debouncer, create_app
from phy.gui.widgets import Barrier
from phy.plot.tests import mouse_click

from ..base import (
    BaseController,
    FeatureMixin,
    TemplateMixin,
    TraceMixin,
    WaveformMixin,
    _allocate_spike_counts,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Mock models and controller classes
# ------------------------------------------------------------------------------


class MyModel:
    seed = np.random.seed(0)
    n_channels = 8
    n_spikes = 20000
    n_clusters = 32
    n_templates = n_clusters
    n_pcs = 5
    n_samples_waveforms = 100
    channel_positions = np.random.normal(size=(n_channels, 2))
    channel_mapping = np.arange(0, n_channels)
    channel_shanks = np.zeros(n_channels, dtype=np.int32)
    features = artificial_features(n_spikes, n_channels, n_pcs)
    metadata = {'group': {3: 'noise', 4: 'mua', 5: 'good'}}
    sample_rate = 10000
    spike_attributes = {}
    amplitudes = np.random.normal(size=n_spikes, loc=1, scale=0.1)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_templates = spike_clusters
    spike_samples = artificial_spike_samples(n_spikes)
    spike_times = spike_samples / sample_rate
    spike_times_reordered = artificial_spike_samples(n_spikes) / sample_rate
    duration = spike_times[-1]
    spike_waveforms = None
    traces = artificial_traces(int(sample_rate * duration), n_channels)

    def __init__(self):
        self.closed = False
        # Clustering mutates this array in place. Keep controller instances independent so
        # actions in one test cannot remove clusters from models created by later tests.
        self.spike_clusters = type(self).spike_clusters.copy()
        self.spike_templates = type(self).spike_templates.copy()

    def _get_some_channels(self, offset, size):
        return list(islice(cycle(range(self.n_channels)), offset, offset + size))

    def get_features(self, spike_ids, channel_ids):
        return artificial_features(len(spike_ids), len(channel_ids), self.n_pcs)

    def get_waveforms(self, spike_ids, channel_ids):
        n_channels = len(channel_ids) if channel_ids else self.n_channels
        return artificial_waveforms(len(spike_ids), self.n_samples_waveforms, n_channels)

    def get_template(self, template_id):
        nc = self.n_channels // 2
        return Bunch(
            template=artificial_waveforms(1, self.n_samples_waveforms, nc)[0, ...],
            channel_ids=self._get_some_channels(template_id, nc),
        )

    def save_spike_clusters(self, spike_clusters):
        pass

    def save_metadata(self, name, values):
        pass

    def close(self):
        self.closed = True


class MyController(BaseController):
    """Default controller."""

    def get_best_channels(self, cluster_id):
        return self.model._get_some_channels(cluster_id, 5)

    def get_channel_amplitudes(self, cluster_id):
        return self.model._get_some_channels(cluster_id, 5), np.ones(5)


class MyControllerW(WaveformMixin, MyController):
    """With waveform view."""


class MyControllerF(FeatureMixin, MyController):
    """With feature view."""


class MyControllerT(TraceMixin, MyController):
    """With trace view."""


class MyControllerTmp(TemplateMixin, MyController):
    """With templates."""


class MyControllerFull(TemplateMixin, WaveformMixin, FeatureMixin, TraceMixin, MyController):
    """With everything."""


def _mock_controller(tempdir, cls):
    model = MyModel()
    return cls(
        dir_path=tempdir,
        config_dir=tempdir / 'config',
        model=model,
        clear_cache=True,
        enable_threading=False,
    )


def test_allocate_spike_counts_redistributes_total_budget():
    np.testing.assert_array_equal(
        _allocate_spike_counts([0, 1, 100], per_cluster=10, total=7),
        [0, 1, 6],
    )
    np.testing.assert_array_equal(
        _allocate_spike_counts([100, 100, 100], per_cluster=10, total=8),
        [3, 3, 2],
    )
    np.testing.assert_array_equal(
        _allocate_spike_counts([100, 2], per_cluster=10, total=None),
        [10, 2],
    )


def test_controller_close(tempdir):
    controller = _mock_controller(tempdir, MyController)
    model = controller.model
    handlers = list(controller._log_handlers)

    assert handlers
    assert all(handler in logging.getLogger('phy').handlers for handler in handlers)

    controller.close()
    controller.close()  # Cleanup is idempotent.

    assert model.closed
    assert all(handler not in logging.getLogger('phy').handlers for handler in handlers)
    assert all(handler.stream is None for handler in handlers)


def test_get_firing_rate_fast_path():
    spike_times = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    clustering = Clustering(np.array([0, 1, 1, 2, 2, 3]))
    controller = object.__new__(BaseController)
    controller.model = Bunch(spike_times=spike_times, duration=4.0)
    controller.supervisor = Bunch(clustering=clustering)

    # Keep a reference selector to compare with the previous implementation.
    reference_selector = SpikeSelector(
        get_spikes_per_cluster=lambda cluster_id: clustering.spikes_per_cluster.get(
            cluster_id, np.array([], dtype=np.int64)
        ),
        spike_times=np.arange(len(spike_times)),
        chunk_bounds=[0, len(spike_times)],
        n_chunks_kept=1,
    )

    def fail_selector(*args, **kwargs):
        raise AssertionError("The firing-rate path must not invoke SpikeSelector.")

    controller.selector = fail_selector

    for cluster_id in (0, 1, 99):
        expected = spike_times[reference_selector(None, [cluster_id])]
        np.testing.assert_array_equal(controller.get_spike_times(cluster_id), expected)
        bunch = controller._get_firing_rate(cluster_id)
        np.testing.assert_array_equal(bunch.data, expected)
        assert bunch.x_min == 0
        assert bunch.x_max == controller.model.duration

    selector_calls = []

    def capped_selector(n, cluster_ids, **kwargs):
        selector_calls.append((n, cluster_ids, kwargs))
        return np.array([2], dtype=np.int64)

    controller.selector = capped_selector
    np.testing.assert_array_equal(controller.get_spike_times(1, n=1), spike_times[[2]])
    assert selector_calls == [(1, [1], {})]

    # The fast path follows live clustering changes rather than stale model assignments.
    controller.selector = fail_selector
    merged = clustering.merge([0, 1])
    expected = spike_times[reference_selector(None, merged.added)]
    bunch = controller._get_firing_rate(merged.added[0])
    np.testing.assert_array_equal(bunch.data, expected)


def test_get_correlograms_rate_fast_path():
    clustering = Clustering(np.array([0, 0, 0, 1, 2, 2]))
    controller = object.__new__(BaseController)
    controller.model = Bunch(duration=4.0)
    controller.supervisor = Bunch(clustering=clustering)
    controller.n_spikes_correlograms = 2
    controller.n_spikes_correlograms_total = None

    def fail_selector(*args, **kwargs):
        raise AssertionError("The correlogram-rate path must not invoke SpikeSelector.")

    controller.selector = fail_selector
    actual = controller._get_correlograms_rate([0, 1, 2, 99], bin_size=0.1)
    counts = np.array([2, 1, 2, 0])
    expected = counts * np.c_[counts] * (0.1 / 4.0)
    np.testing.assert_array_equal(actual, expected)

    controller.n_spikes_correlograms = 10
    controller.n_spikes_correlograms_total = 4
    actual = controller._get_correlograms_rate([0, 1, 2], bin_size=0.1)
    counts = np.array([2, 1, 1])
    expected = counts * np.c_[counts] * (0.1 / 4.0)
    np.testing.assert_array_equal(actual, expected)

    controller.n_spikes_correlograms = None
    controller.n_spikes_correlograms_total = None
    actual = controller._get_correlograms_rate([0, 1, 2], bin_size=0.1)
    counts = np.array([3, 1, 2])
    expected = counts * np.c_[counts] * (0.1 / 4.0)
    np.testing.assert_array_equal(actual, expected)


def test_correlogram_cache_key_includes_spike_limit(tempdir):
    controller = _mock_controller(tempdir, MyController)
    selector = controller.selector
    calls = []

    def recording_selector(n, cluster_ids, **kwargs):
        calls.append(n)
        return selector(n, cluster_ids, **kwargs)

    controller.selector = recording_selector
    controller.n_spikes_correlograms = 1
    controller._get_correlograms([0], bin_size=0.001, window_size=0.05)
    controller.n_spikes_correlograms = 2
    controller._get_correlograms([0], bin_size=0.001, window_size=0.05)
    # The identical request at the same limit should use the disk cache.
    controller._get_correlograms([0], bin_size=0.001, window_size=0.05)

    assert calls == [1, 2]

    controller.n_spikes_correlograms = 10
    controller.n_spikes_correlograms_total = 1
    controller._get_correlograms([0], bin_size=0.002, window_size=0.05)
    controller.n_spikes_correlograms_total = 2
    controller._get_correlograms([0], bin_size=0.002, window_size=0.05)
    controller._get_correlograms([0], bin_size=0.002, window_size=0.05)

    # Changing either limit creates a distinct cache entry.
    assert calls == [1, 2, 1, 2]


def test_correlogram_sampling_preserves_nearby_pairs():
    n_spikes = 100_000
    spike_times = np.arange(n_spikes, dtype=float) / 1000.0
    clustering = Clustering(np.zeros(n_spikes, dtype=np.int32))
    controller = object.__new__(BaseController)
    controller.model = Bunch(
        spike_times=spike_times,
        spike_samples=np.arange(n_spikes, dtype=np.int64),
        sample_rate=1000.0,
        duration=spike_times[-1],
    )
    controller.supervisor = Bunch(clustering=clustering)
    controller.n_spikes_correlograms = 1000
    controller.selector = SpikeSelector(
        get_spikes_per_cluster=lambda cluster_id: clustering.spikes_per_cluster[cluster_id],
        spike_times=controller.model.spike_samples,
        chunk_bounds=[0, n_spikes],
        n_chunks_kept=1,
    )

    np.random.seed(0)
    correlogram = controller._get_correlograms([0], bin_size=.001, window_size=.05)
    assert correlogram[0, 0].sum() > 0


def test_sparse_waveform_selection_filters_small_exported_pool(tempdir):
    controller = _mock_controller(tempdir, MyControllerW)
    subset_spikes = np.arange(0, controller.model.n_spikes, 2, dtype=np.int64)
    controller.model.spike_waveforms = Bunch(spike_ids=subset_spikes)
    selected = []
    get_waveforms = controller.model.get_waveforms

    def fail_selector(*args, **kwargs):
        raise AssertionError("sparse waveform selection scanned a full cluster")

    def capture_waveforms(spike_ids, channel_ids):
        selected.append(spike_ids)
        return get_waveforms(spike_ids, channel_ids)

    controller.selector = fail_selector
    controller.model.get_waveforms = capture_waveforms
    bunch = controller._get_waveforms_with_n_spikes(0, 3)
    assert bunch.data.shape[0] == 3
    eligible = subset_spikes[
        controller.supervisor.clustering.spike_clusters[subset_spikes] == 0
    ]
    np.testing.assert_array_equal(selected[0], _sample_spikes_evenly(eligible, 3))


def test_waveform_selected_clusters_share_total_budget(tempdir):
    controller = _mock_controller(tempdir, MyControllerW)
    controller.n_spikes_waveforms = 100
    controller.n_spikes_waveforms_total = 10

    counts = [
        controller._get_waveform_spike_count(cluster_id, cluster_ids=[0, 1, 2])
        for cluster_id in [0, 1, 2]
    ]

    assert counts == [4, 3, 3]


def test_amplitude_background_has_stable_total_budget(tempdir):
    controller = _mock_controller(tempdir, MyControllerTmp)
    controller.n_spikes_amplitudes = 7
    controller.n_spikes_amplitudes_background = 7
    selected_cluster = 0

    data = controller._amplitude_getter([selected_cluster, None], name='template')
    repeat = controller._amplitude_getter([selected_cluster, None], name='template')
    selected, background = data

    # The selected cluster retains its independent display budget, while all
    # grey background clusters share one fixed budget.
    assert len(selected.spike_ids) == controller.n_spikes_amplitudes
    assert len(background.spike_ids) == controller.n_spikes_amplitudes_background
    np.testing.assert_array_equal(background.spike_ids, repeat[1].spike_ids)
    assert np.all(np.diff(background.spike_ids) >= 0)

    other_clusters = set(controller.get_clusters_on_channel(0)) - {selected_cluster}
    background_clusters = controller.supervisor.clustering.spike_clusters[background.spike_ids]
    assert set(background_clusters).issubset(other_clusters)

    # Lasso/split requests must still retrieve every eligible spike.
    all_data = controller._amplitude_getter(
        [selected_cluster, None], name='template', load_all=True
    )
    expected_background = sum(
        len(controller.supervisor.clustering.spikes_per_cluster[cluster_id])
        for cluster_id in other_clusters
    )
    assert len(all_data[0].spike_ids) == len(
        controller.supervisor.clustering.spikes_per_cluster[selected_cluster]
    )
    assert len(all_data[1].spike_ids) == expected_background


def test_amplitude_selected_clusters_share_total_budget(tempdir):
    controller = _mock_controller(tempdir, MyControllerTmp)
    controller.n_spikes_amplitudes = 7
    controller.n_spikes_amplitudes_total = 10
    controller.n_spikes_amplitudes_background = 7

    selected_a, selected_b, background = controller._amplitude_getter(
        [0, 1, None], name='template'
    )

    assert len(selected_a.spike_ids) == 5
    assert len(selected_b.spike_ids) == 5
    assert len(background.spike_ids) == 7


def test_amplitude_background_redistributes_unused_budget():
    # Cluster 0 is empty, cluster 1 has one spike, and cluster 2 has ample
    # capacity. The background budget should be filled while retaining the
    # small nonempty cluster's representation.
    controller = object.__new__(BaseController)
    controller.supervisor = Bunch(clustering=Clustering(np.array([1] + [2] * 100, dtype=np.int64)))

    spike_ids = controller._get_background_amplitude_spike_ids([0, 1, 2], n=5)

    assert len(spike_ids) == 5
    assert len(np.unique(spike_ids)) == 5
    assert np.all(np.diff(spike_ids) >= 0)
    assert 0 in spike_ids


def test_get_firing_rate_honors_get_spike_times_override():
    class OverrideController(BaseController):
        def get_spike_times(self, cluster_id, n=None):
            assert cluster_id == 7
            return np.array([1.25, 2.5])

    controller = object.__new__(OverrideController)
    controller.model = Bunch(duration=3.0)
    bunch = controller._get_firing_rate(7)
    np.testing.assert_array_equal(bunch.data, [1.25, 2.5])
    assert bunch.x_min == 0
    assert bunch.x_max == 3.0


def test_amplitude_view_excludes_unavailable_features(qtbot, tempdir):
    controller = _mock_controller(tempdir, MyControllerFull)
    controller.model.features = None
    try:
        view = controller.create_amplitude_view()
        assert list(view.amplitudes) == ['template']
        view.amplitudes_type = 'feature'
        assert view.amplitudes_type == 'template'
    finally:
        controller.close()


# ------------------------------------------------------------------------------
# Base classes
# ------------------------------------------------------------------------------


class MinimalControllerTests:
    # Methods to override
    # --------------------------------------------------------------------------

    @classmethod
    def get_controller(cls, tempdir):
        raise NotImplementedError()

    # Convenient properties
    # --------------------------------------------------------------------------

    @property
    def qtbot(self):
        return self.__class__._qtbot

    @property
    def controller(self):
        return self.__class__._controller

    @property
    def model(self):
        return self.__class__._controller.model

    @property
    def supervisor(self):
        return self.controller.supervisor

    @property
    def cluster_view(self):
        return self.supervisor.cluster_view

    @property
    def similarity_view(self):
        return self.supervisor.similarity_view

    @property
    def cluster_ids(self):
        return self.supervisor.clustering.cluster_ids

    @property
    def gui(self):
        return self.__class__._gui

    @property
    def selected(self):
        return self.supervisor.selected

    @property
    def amplitude_view(self):
        return self.gui.list_views(AmplitudeView)[0]

    # Convenience methods
    # --------------------------------------------------------------------------

    def stop(self):  # pragma: no cover
        """Used for debugging."""
        create_app().exec_()
        self.gui.close()

    def next(self):
        s = self.supervisor
        s.select_actions.next()
        s.block()

    def next_best(self):
        s = self.supervisor
        s.select_actions.next_best()
        s.block()

    def label(self, name, value):
        s = self.supervisor
        s.actions.label(name, value)
        s.block()

    def merge(self):
        s = self.supervisor
        s.actions.merge()
        s.block()

    def split(self):
        s = self.supervisor
        s.actions.split()
        s.block()

    def undo(self):
        s = self.supervisor
        s.actions.undo()
        s.block()

    def redo(self):
        s = self.supervisor
        s.actions.redo()
        s.block()

    def move(self, w):
        s = self.supervisor
        getattr(s.actions, f'move_{w}')()
        s.block()

    def lasso(self, view, scale=1.0):
        w, h = view.canvas.get_size()
        w *= scale
        h *= scale
        mouse_click(self.qtbot, view.canvas, (1, 1), modifiers=('Control',))
        mouse_click(self.qtbot, view.canvas, (w - 1, 1), modifiers=('Control',))
        mouse_click(self.qtbot, view.canvas, (w - 1, h - 1), modifiers=('Control',))
        mouse_click(self.qtbot, view.canvas, (1, h - 1), modifiers=('Control',))

    # Fixtures
    # --------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        Debouncer.delay = 1
        cls._qtbot = QtBot(create_app())
        cls._tempdir_ = tempfile.mkdtemp()
        cls._tempdir = Path(cls._tempdir_)
        cls._controller = cls.get_controller(cls._tempdir)
        cls._create_gui()

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
            cls._qtbot.stop()
        cls._close_gui()
        cls._controller.close()
        shutil.rmtree(cls._tempdir_)

    @classmethod
    def _create_gui(cls):
        cls._gui = cls._controller.create_gui(do_prompt_save=False)
        s = cls._controller.supervisor
        b = Barrier()
        connect(b('cluster_view'), event='ready', sender=s.cluster_view)
        connect(b('similarity_view'), event='ready', sender=s.similarity_view)
        with cls._qtbot.waitExposed(cls._gui):
            cls._gui.show()
        # cls._qtbot.addWidget(cls._gui)
        b.wait()

    @classmethod
    def _close_gui(cls):
        cls._gui.close()
        cls._gui.deleteLater()
        cls._qtbot.wait(100)

        # NOTE: make sure all callback functions are unconnected at the end of the tests
        # to avoid side-effects and spurious dependencies between tests.
        reset()


class BaseControllerTests(MinimalControllerTests):
    # Common test methods
    # --------------------------------------------------------------------------

    def test_common_01(self):
        """Select one cluster."""
        self.supervisor.select_actions.reset_wizard()
        self.supervisor.block()
        self.next_best()
        self.assertEqual(len(self.selected), 1)

    def test_common_02(self):
        """Select one similar cluster."""
        self.next()
        self.assertEqual(len(self.selected), 2)

    def test_common_03(self):
        """Select another similar cluster."""
        self.next()
        self.assertEqual(len(self.selected), 2)

    def test_common_04(self):
        """Merge the selected clusters."""
        self.merge()
        self.assertEqual(len(self.selected), 1)

    def test_common_05(self):
        """Select a similar cluster."""
        self.next()
        self.assertEqual(len(self.selected), 2)

    def test_common_06(self):
        """Undo/redo the merge several times."""
        for _ in range(3):
            self.undo()
            self.assertEqual(len(self.selected), 2)

            self.redo()
            self.assertEqual(len(self.selected), 2)

    def test_common_07(self):
        """Move action."""
        self.move('similar_to_noise')
        self.assertEqual(len(self.selected), 2)

    def test_common_08(self):
        """Move action."""
        self.move('best_to_good')
        self.assertEqual(len(self.selected), 1)

    def test_common_09(self):
        """Label action."""
        self.next()

        @connect(sender=self.supervisor)
        def on_cluster(sender, up):
            cls = self.__class__
            cls._label_name, cls._label_value = 'new_label', up.metadata_value

        self.label('new_label', 3)

        unconnect(on_cluster)

    def test_common_10(self):
        self.supervisor.save()

    def test_common_11(self):
        s = self.controller.selection
        self.assertEqual(s.cluster_ids, self.selected)
        self.gui.view_actions.toggle_spike_reorder(True)
        self.gui.view_actions.switch_raw_data_filter()


class GlobalViewsTests:
    def test_global_filter_1(self):
        self.next()
        cv = self.supervisor.cluster_view
        emit('table_filter', cv, self.cluster_ids[::2])

    def test_global_sort_1(self):
        cv = self.supervisor.cluster_view
        emit('table_sort', cv, self.cluster_ids[::-1])


# ------------------------------------------------------------------------------
# Mock test cases
# ------------------------------------------------------------------------------


class MockControllerTests(MinimalControllerTests, GlobalViewsTests, unittest.TestCase):
    """Empty mock controller."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyController)

    @mark.filterwarnings(
        'ignore:Parsing dates involving a day of month without a year specified is ambiguious:DeprecationWarning'
    )
    def test_create_ipython_view(self):
        view = self.gui.create_and_add_view('IPythonView')
        view.stop()
        view.dock.close()
        self.qtbot.wait(100)

    def test_create_raster_view(self):
        view = self.gui.create_and_add_view('RasterView')
        mouse_click(self.qtbot, view.canvas, (10, 10), modifiers=('Control',))
        view.actions.next_color_scheme()


class MockControllerWTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with waveforms."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerW)

    @property
    def waveform_view(self):
        return self.gui.list_views(WaveformView)[0]

    def test_waveform_view(self):
        self.waveform_view.actions.toggle_mean_waveforms(True)
        self.waveform_view.actions.next_waveforms_type()
        self.waveform_view.actions.change_n_spikes_waveforms(200)

    def test_mean_amplitudes(self):
        self.next()
        self.assertTrue(self.controller.get_mean_spike_raw_amplitudes(self.selected[0]) >= 0)

    def test_waveform_select_channel(self):
        self.amplitude_view.amplitudes_type = 'raw'

        fv = self.waveform_view
        # Select channel in waveform view.
        w, h = fv.canvas.get_size()
        w, h = w / 2, h / 2
        x, y = w / 2, h / 2
        mouse_click(self.qtbot, fv.canvas, (x, y), button='Left', modifiers=('Control',))


class MockControllerFTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with features."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerF)

    @property
    def feature_view(self):
        return self.gui.list_views(FeatureView)[0]

    def test_feature_view_split(self):
        self.next()
        n = max(self.cluster_ids)
        self.lasso(self.feature_view, 0.1)
        self.split()
        # Split one cluster => Two new clusters should be selected after the split.
        self.assertEqual(self.selected[:2], [n + 1, n + 2])

    def test_feature_view_toggle_spike_reorder(self):
        self.gui.view_actions.toggle_spike_reorder(True)

    def test_select_feature(self):
        self.next()

        fv = self.feature_view
        # Select feature in feature view.
        w, h = fv.canvas.get_size()
        w, h = w / 4, h / 4
        x, y = w / 2, h / 2
        mouse_click(self.qtbot, fv.canvas, (x, y), button='Right', modifiers=('Alt',))


class MockControllerTTests(GlobalViewsTests, MinimalControllerTests, unittest.TestCase):
    """Mock controller with traces."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerT)

    @property
    def trace_view(self):
        return self.gui.list_views(TraceView)[0]

    def test_trace_view(self):
        self.trace_view.actions.go_to_next_spike()
        self.trace_view.actions.go_to_previous_spike()
        self.trace_view.actions.toggle_highlighted_spikes(True)
        mouse_click(self.qtbot, self.trace_view.canvas, (100, 100), modifiers=('Control',))
        mouse_click(self.qtbot, self.trace_view.canvas, (150, 100), modifiers=('Shift',))
        emit('select_time', self, 0)
        self.trace_view.actions.next_color_scheme()


class MockControllerTmpTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with templates."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerTmp)

    @property
    def template_view(self):
        return self.gui.list_views(TemplateView)[0]

    def test_template_view_select(self):
        mouse_click(self.qtbot, self.template_view.canvas, (100, 100), modifiers=('Control',))
        mouse_click(self.qtbot, self.template_view.canvas, (150, 100), modifiers=('Shift',))

    def test_mean_amplitudes(self):
        self.next()
        self.assertTrue(self.controller.get_mean_spike_template_amplitudes(self.selected[0]) >= 0)

    def test_split_template_amplitude(self):
        self.next()
        self.amplitude_view.amplitudes_type = 'template'
        self.controller.get_amplitudes(self.selected[0], load_all=True)
        self.amplitude_view.plot()
        self.lasso(self.amplitude_view)
        self.split()


class MockControllerFullTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with all views."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerFull)

    def test_filter(self):
        rdf = self.controller.raw_data_filter

        @rdf.add_filter
        def diff(arr, axis=0):  # pragma: no cover
            out = np.zeros_like(arr)
            if axis == 0:
                out[1:, ...] = np.diff(arr, axis=axis)
            elif axis == 1:
                out[:, 1:, ...] = np.diff(arr, axis=axis)
            return out

        self.gui.view_actions.switch_raw_data_filter()
        self.gui.view_actions.switch_raw_data_filter()

        rdf.set('diff')
        assert rdf.current == 'diff'

    def test_y1_close_view(self):
        s = self.selected
        self.next_best()
        assert s != self.selected
        fv = self.gui.get_view(FeatureView)
        wv = self.gui.get_view(WaveformView)
        assert self.selected == wv.cluster_ids
        fv.dock.close()
        s = self.selected
        self.next_best()
        assert s != self.selected
        assert self.selected == wv.cluster_ids

    def test_z1_close_all_views(self):
        self.next()

        for view in self.gui.views:
            view.dock.close()
            self.qtbot.wait(200)

    def test_z2_open_all_views(self):
        for view_cls in self.controller.view_creator.keys():
            self.gui.create_and_add_view(view_cls)
            self.qtbot.wait(200)

    def test_z3_select(self):
        self.next()
        self.next()

    def test_z4_open_new_views(self):
        for view_cls in self.controller.view_creator.keys():
            self.gui.create_and_add_view(view_cls)
            self.qtbot.wait(200)

    def test_z5_select(self):
        self.next_best()
        self.next()
