"""Integration tests for the GUIs."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import json
import logging
import os
import shutil
import tempfile
import unittest
from itertools import cycle, islice
from pathlib import Path
from unittest.mock import patch

import numpy as np
from phylib.io.array import SpikeSelector
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

from phy.cluster._propositions import PropositionStatus
from phy.cluster.clustering import Clustering
from phy.cluster.views import (
    AmplitudeView,
    CorrelogramView,
    FeatureView,
    FiringRateView,
    TemplateView,
    TraceView,
    WaveformView,
)
from phy.gui import GUI
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
    _select_spikes_evenly,
    _spike_budget_fields,
    _spike_budget_values,
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


class MyPropositionController(MyController):
    enable_merge_propositions = True


def _mock_controller(tempdir, cls):
    model = MyModel()
    return cls(
        dir_path=tempdir,
        config_dir=tempdir / 'config',
        model=model,
        clear_cache=True,
        enable_threading=False,
    )


def test_controller_loads_and_reopens_merge_proposition_reviews(tempdir):
    source = {
        'format_version': '2',
        'unit_ids': list(range(MyModel.n_clusters)),
        'merges': [{'unit_ids': [1, 2]}],
    }
    (tempdir / 'curation.json').write_text(json.dumps(source), encoding='utf8')
    controller = _mock_controller(tempdir, MyPropositionController)
    try:
        proposition = controller.supervisor.merge_propositions.catalog.propositions[0]

        controller.supervisor.merge_propositions.reject(proposition.key)
        controller.supervisor.save()

        sidecar = json.loads((tempdir / 'curation_review.json').read_text(encoding='utf8'))
        assert sidecar['source']['filename'] == 'curation.json'
        assert len(sidecar['source']['sha256']) == 64
        assert sidecar['reviews'][proposition.key]['decision'] == 'rejected'
    finally:
        controller.close()

    reopened = _mock_controller(tempdir, MyPropositionController)
    try:
        assert (
            reopened.supervisor.merge_propositions.catalog.status_for(proposition.key)
            is PropositionStatus.REJECTED
        )
    finally:
        reopened.close()


def test_invalid_curation_json_does_not_prevent_ordinary_controller(tempdir, caplog):
    (tempdir / 'curation.json').write_text('{bad', encoding='utf8')

    controller = _mock_controller(tempdir, MyPropositionController)
    try:
        assert controller.supervisor.merge_propositions is None
        assert 'Merge Propositions disabled' in caplog.text
    finally:
        controller.close()


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
    assert WaveformMixin.n_spikes_waveforms_total is None
    np.testing.assert_array_equal(
        _allocate_spike_counts(
            [100, 100, 100],
            per_cluster=WaveformMixin.n_spikes_waveforms,
            total=WaveformMixin.n_spikes_waveforms_total,
        ),
        [100, 100, 100],
    )
    assert BaseController.n_spikes_amplitudes_total is None
    assert BaseController.n_spikes_correlograms_total is None


def test_spike_budget_dialog_supports_independent_optional_limits():
    fields = _spike_budget_fields(None, 400, max_n_clusters=8)
    defaults = {field['name']: field['default'] for field in fields}
    assert defaults['use_per_cluster'] is False
    assert defaults['use_total'] is True
    assert _spike_budget_values(
        {
            'use_per_cluster': True,
            'per_cluster': 100,
            'use_total': False,
            'total': 400,
        }
    ) == (100, None)


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


def test_set_selector_supports_released_and_newer_phylib():
    controller = object.__new__(BaseController)
    controller.model = Bunch(spike_samples=np.arange(4))
    controller.supervisor = Bunch(clustering=Clustering(np.array([0, 0, 1, 1])))
    controller.n_chunks_kept = 2

    class ReleasedSpikeSelector:
        def __init__(
            self,
            get_spikes_per_cluster=None,
            spike_times=None,
            chunk_bounds=None,
            n_chunks_kept=None,
        ):
            self.spikes_are_disjoint = None

    with patch('phy.apps.base.SpikeSelector', ReleasedSpikeSelector):
        controller._set_selector()
    assert controller.selector.spikes_are_disjoint is None

    class NewerSpikeSelector(ReleasedSpikeSelector):
        def __init__(self, *args, spikes_are_disjoint=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.spikes_are_disjoint = spikes_are_disjoint

    with patch('phy.apps.base.SpikeSelector', NewerSpikeSelector):
        controller._set_selector()
    assert controller.selector.spikes_are_disjoint is True


def test_select_spikes_evenly_supports_released_and_newer_phylib():
    spikes = {0: np.arange(5), 1: np.arange(5, 10)}

    def released_selector(n_spikes, cluster_ids, subset_chunks=False):
        assert subset_chunks
        assert n_spikes is None
        return np.concatenate([spikes[cluster_id] for cluster_id in cluster_ids])

    np.testing.assert_array_equal(
        _select_spikes_evenly(released_selector, 2, [0, 1], subset_chunks=True),
        [0, 4, 5, 9],
    )

    def newer_selector(n_spikes, cluster_ids, subset_chunks=False, sample_evenly=False):
        assert (n_spikes, cluster_ids, subset_chunks, sample_evenly) == (
            2,
            [0, 1],
            True,
            True,
        )
        return np.array([1, 8])

    np.testing.assert_array_equal(
        _select_spikes_evenly(newer_selector, 2, [0, 1], subset_chunks=True),
        [1, 8],
    )


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
        raise AssertionError('The firing-rate path must not invoke SpikeSelector.')

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
    assert controller.n_spikes_correlograms_total is None

    def fail_selector(*args, **kwargs):
        raise AssertionError('The correlogram-rate path must not invoke SpikeSelector.')

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
    controller.close()


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
    correlogram = controller._get_correlograms([0], bin_size=0.001, window_size=0.05)
    assert correlogram[0, 0].sum() > 0


def test_sparse_waveform_selection_filters_small_exported_pool(tempdir):
    controller = _mock_controller(tempdir, MyControllerW)
    subset_spikes = np.arange(0, controller.model.n_spikes, 2, dtype=np.int64)
    controller.model.spike_waveforms = Bunch(spike_ids=subset_spikes)
    selected = []
    get_waveforms = controller.model.get_waveforms

    def fail_selector(*args, **kwargs):
        raise AssertionError('sparse waveform selection scanned a full cluster')

    def capture_waveforms(spike_ids, channel_ids):
        selected.append(spike_ids)
        return get_waveforms(spike_ids, channel_ids)

    controller.selector = fail_selector
    controller.model.get_waveforms = capture_waveforms
    bunch = controller._get_waveforms_with_n_spikes(0, 3)
    assert bunch.data.shape[0] == 3
    eligible = subset_spikes[controller.supervisor.clustering.spike_clusters[subset_spikes] == 0]
    expected_indices = [0, (len(eligible) - 1) // 2, len(eligible) - 1]
    np.testing.assert_array_equal(selected[0], eligible[expected_indices])
    np.testing.assert_array_equal(bunch.spike_ids, selected[0])
    controller.close()


def test_mean_waveforms_do_not_expose_individual_spike_ids(tempdir):
    controller = _mock_controller(tempdir, MyControllerW)
    bunch = controller._get_mean_waveforms(0)
    assert bunch.data.shape[0] == 1
    assert 'spike_ids' not in bunch
    controller.close()


def test_amplitude_preview_highlights_waveforms_with_cached_resolver(qtbot, tempdir):
    controller = _mock_controller(tempdir, MyControllerW)
    amplitude = controller.create_amplitude_view()
    waveform = controller.create_waveform_view()
    gui = GUI(name='AmplitudePreview', config_dir=tempdir)
    amplitude.attach(gui)
    waveform.attach(gui)
    waveform.on_select(cluster_ids=[0])
    spike_ids = waveform._displayed_bunchs[0].spike_ids
    calls = []

    def resolve(ids, name, first_cluster):
        calls.append((ids.copy(), name, first_cluster))
        return np.arange(len(ids), dtype=float)

    controller._resolve_spike_amplitudes = resolve
    emit(
        'amplitude_split_preview_changed',
        amplitude,
        cluster_id=0,
        amplitudes_type='raw',
        threshold=1.5,
    )
    np.testing.assert_array_equal(waveform._highlighted_spike_ids, spike_ids[:2])
    assert waveform._highlighted_spike_color == amplitude.split_preview_color
    assert len(calls) == 1

    # Moving only the threshold must reuse the amplitudes already resolved for
    # the displayed waveform identities.
    emit(
        'amplitude_split_preview_changed',
        amplitude,
        cluster_id=0,
        amplitudes_type='raw',
        threshold=2.5,
    )
    np.testing.assert_array_equal(waveform._highlighted_spike_ids, spike_ids[:3])
    assert len(calls) == 1

    amplitude.split_threshold = 1.5
    emit('selected_channel_changed', waveform)
    assert amplitude.split_threshold is None

    emit(
        'amplitude_split_preview_changed',
        amplitude,
        cluster_id=1,
        amplitudes_type='raw',
        threshold=2.5,
    )
    assert not len(waveform._highlighted_spike_ids)

    amplitude.split_threshold = 0.5
    emit(
        'amplitude_split_preview_changed',
        amplitude,
        cluster_id=0,
        amplitudes_type='raw',
        threshold=amplitude.split_threshold,
    )
    assert len(waveform._highlighted_spike_ids)
    amplitude.dock.close()
    assert not len(waveform._highlighted_spike_ids)
    waveform.dock.close()
    gui.close()
    controller.close()


def test_amplitude_threshold_split_commits_exact_partition_and_undo_redo(qtbot, tempdir):
    """The sampled preview must commit the exact, all-spike threshold partition."""
    controller = _mock_controller(tempdir, MyControllerFull)
    controller.n_spikes_amplitudes = 3
    controller.n_spikes_waveforms = 8
    supervisor = controller.supervisor
    cluster_id = 0
    original_clusters = controller.model.spike_clusters.copy()
    cluster_spike_ids = np.flatnonzero(original_clusters == cluster_id)
    # The sparse display samples cannot be authoritative: this deterministic
    # pattern gives both sides of the threshold throughout the cluster.
    controller.model.amplitudes = np.full(controller.model.n_spikes, 3.0)
    controller.model.amplitudes[cluster_spike_ids] = np.arange(len(cluster_spike_ids)) % 4
    expected = cluster_spike_ids[controller.model.amplitudes[cluster_spike_ids] < 1.5]
    remaining = np.setdiff1d(cluster_spike_ids, expected)
    gui = controller.create_gui(do_prompt_save=False)
    try:
        with qtbot.waitExposed(gui):
            gui.show()
        supervisor.select([cluster_id])
        supervisor.block()
        amplitude = gui.list_views(AmplitudeView)[0]
        waveform = gui.list_views(WaveformView)[0]
        amplitude.amplitudes_type = 'template'
        amplitude.plot()
        waveform.waveforms_type = 'waveforms'
        waveform.plot()

        displayed_amplitudes = next(
            bunch for bunch in amplitude._displayed_bunchs if bunch.cluster_id == cluster_id
        )
        waveform_spike_ids = waveform._displayed_bunchs[0].spike_ids
        missing_from_amplitude = np.setdiff1d(waveform_spike_ids, displayed_amplitudes.spike_ids)
        assert len(missing_from_amplitude)
        assert np.any(np.isin(expected, missing_from_amplitude))

        # Set the transient threshold through the view-level preview contract;
        # waveform classification resolves its own displayed spike identities.
        amplitude.split_threshold = 1.5
        amplitude._replot_displayed_amplitudes()
        emit(
            'amplitude_split_preview_changed',
            amplitude,
            cluster_id=cluster_id,
            amplitudes_type=amplitude.amplitudes_type,
            threshold=amplitude.split_threshold,
        )
        expected_waveform = waveform_spike_ids[
            controller.model.amplitudes[waveform_spike_ids] < amplitude.split_threshold
        ]
        np.testing.assert_array_equal(waveform._highlighted_spike_ids, expected_waveform)

        # This is the same request_split path invoked by the K shortcut.
        supervisor.actions.split()
        supervisor.block()
        after_split = controller.model.spike_clusters.copy()
        split_cluster = after_split[expected]
        remaining_cluster = after_split[remaining]
        assert len(np.unique(split_cluster)) == len(np.unique(remaining_cluster)) == 1
        assert split_cluster[0] != remaining_cluster[0]
        assert not np.any(after_split[cluster_spike_ids] == cluster_id)
        assert amplitude.split_threshold is None
        assert not len(waveform._highlighted_spike_ids)

        supervisor.actions.undo()
        supervisor.block()
        np.testing.assert_array_equal(controller.model.spike_clusters, original_clusters)
        assert amplitude.split_threshold is None

        supervisor.actions.redo()
        supervisor.block()
        redone = controller.model.spike_clusters
        np.testing.assert_array_equal(redone[expected], split_cluster)
        np.testing.assert_array_equal(redone[remaining], remaining_cluster)
    finally:
        gui.close()
        controller.close()


def test_waveform_selected_clusters_share_total_budget(tempdir):
    controller = _mock_controller(tempdir, MyControllerW)
    controller.n_spikes_waveforms = 100
    controller.n_spikes_waveforms_total = 10

    counts = [
        controller._get_waveform_spike_count(cluster_id, cluster_ids=[0, 1, 2])
        for cluster_id in [0, 1, 2]
    ]

    assert counts == [4, 3, 3]
    controller.close()


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
    controller.close()


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
    controller.close()


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


def test_recording_time_unit_updates_compatible_views(qtbot):
    controller = object.__new__(BaseController)
    controller.recording_time_unit = 's'
    controller.recording_time_decimals = 2
    amplitude = AmplitudeView(amplitudes=lambda cluster_ids, load_all=False: None)
    firing_rate = FiringRateView(cluster_stat=lambda cluster_id: Bunch(data=np.array([0.0])))

    class GUI:
        views = [amplitude, firing_rate]

    with (
        patch.object(amplitude.canvas, 'update') as amplitude_update,
        patch.object(firing_rate.canvas, 'update') as firing_rate_update,
    ):
        controller._set_recording_time_unit('hours', GUI())

    assert controller.recording_time_unit == 'h'
    assert amplitude.recording_time_unit == firing_rate.recording_time_unit == 'h'
    assert all(label.endswith(' h') for label in amplitude.canvas.axes.locator.xtext)
    assert all(label.endswith(' h') for label in firing_rate.canvas.axes.locator.xtext)
    amplitude_update.assert_called()
    firing_rate_update.assert_called()

    amplitude.close()
    firing_rate.close()


def test_recording_time_unit_menu(qtbot, tempdir):
    controller = object.__new__(BaseController)
    controller.recording_time_unit = 's'
    controller.recording_time_decimals = 2
    gui = GUI(name='RecordingTimeTest', config_dir=tempdir)
    controller.create_misc_actions(gui)

    seconds = gui.view_actions.get('Seconds')
    minutes = gui.view_actions.get('Minutes')
    hours = gui.view_actions.get('Hours')
    assert seconds.isCheckable() and minutes.isCheckable() and hours.isCheckable()
    assert seconds.isChecked()

    hours.trigger()

    assert controller.recording_time_unit == 'h'
    assert hours.isChecked()
    assert not seconds.isChecked()
    assert not minutes.isChecked()
    assert gui.view_actions.get('2 decimals') is None
    gui.close()


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

    def test_correlogram_view_settings(self):
        view = self.gui.list_views(CorrelogramView)[0]
        values = {
            'use_per_cluster': True,
            'per_cluster': 1234,
            'use_total': False,
            'total': 5678,
            'bin_size': 2.0,
            'window_size': 100.0,
            'refractory_period': 3.0,
        }
        with patch('phy.apps.base.view_settings_dialog', return_value=values):
            view.actions.get('View settings').trigger()
        self.assertEqual(self.controller.n_spikes_correlograms, 1234)
        self.assertIsNone(self.controller.n_spikes_correlograms_total)
        self.assertEqual(view.bin_size, 0.002)
        self.assertEqual(view.window_size, 0.1)
        self.assertEqual(view.refractory_period, 0.003)
        self.controller.n_spikes_correlograms = 100000
        self.controller.n_spikes_correlograms_total = None


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

    def test_waveform_view_settings(self):
        values = {
            'use_per_cluster': True,
            'per_cluster': 321,
            'use_total': True,
            'total': 654,
        }
        with patch('phy.apps.base.view_settings_dialog', return_value=values):
            self.waveform_view.actions.get('View settings').trigger()
        self.assertEqual(self.controller.n_spikes_waveforms, 321)
        self.assertEqual(self.controller.n_spikes_waveforms_total, 654)
        self.controller.n_spikes_waveforms = 100
        self.controller.n_spikes_waveforms_total = None

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

    def test_amplitude_view_settings(self):
        values = {
            'use_per_cluster': True,
            'per_cluster': 123,
            'use_total': True,
            'total': 456,
            'background': 789,
        }
        with patch('phy.apps.base.view_settings_dialog', return_value=values):
            self.amplitude_view.actions.get('View settings').trigger()
        self.assertEqual(self.controller.n_spikes_amplitudes, 123)
        self.assertEqual(self.controller.n_spikes_amplitudes_total, 456)
        self.assertEqual(self.controller.n_spikes_amplitudes_background, 789)
        self.controller.n_spikes_amplitudes = 10000
        self.controller.n_spikes_amplitudes_total = None
        self.controller.n_spikes_amplitudes_background = 10000

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
