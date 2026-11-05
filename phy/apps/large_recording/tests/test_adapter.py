"""Bounded GUI correctness and ordinary-controller isolation."""

import json

import numpy as np
import pytest
from click.testing import CliRunner
from phylib.utils import connect, unconnect

from phy.apps import phycli
from phy.apps.large_recording.adapter import BoundedTemplateController
from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.tests.test_source import make_sorting
from phy.apps.template.gui import TemplateController


def make_controller(tmp_path, sparse=False, propositions=None):
    params = make_gui_sorting(tmp_path / 'source')
    if propositions is not None:
        (params.parent / 'curation.json').write_text(json.dumps(propositions))
    output = prepare_large(
        params,
        tmp_path / 'bounded',
        window_seconds=3 if sparse else 20,
        stratum_seconds=20,
        max_spikes=5 if sparse else 100,
        bin_seconds=5,
        batch_size=2,
    )
    controller = TemplateController(
        dir_path=output,
        config_dir=tmp_path / 'config',
        plugins=[],
        enable_threading=False,
        clear_cache=True,
    )
    return params, controller


def make_gui_sorting(path):
    params = make_sorting(path)
    times = np.load(path / 'spike_times.npy')
    np.save(path / 'spike_times.npy', times * 30000)
    params.write_text(
        params.read_text()
        .replace('sample_rate = 1.0', 'sample_rate = 30000.0')
        .replace('num_samples = 20', 'num_samples = 600000')
    )
    return params


def test_manifest_factory_reserves_complete_ids_and_guards_mutations(tmp_path):
    _, controller = make_controller(tmp_path, sparse=True)
    try:
        assert isinstance(controller, BoundedTemplateController)
        with pytest.raises(ValueError):
            controller.model.spike_clusters[0] = 123
        with pytest.raises(ValueError):
            controller.model.spike_clusters = np.array([123])
        with pytest.raises(ValueError):
            controller.model.sample_rate = 1.0
        with pytest.raises(TypeError):
            controller.model.metadata['group'][0] = 'noise'
        supervisor = controller.supervisor
        assert 99 not in supervisor.clustering.cluster_ids
        assert supervisor.clustering.new_cluster_id() == 100
        assert controller.get_full_count(0) == 4
        assert controller.get_mean_firing_rate(0) == 0.2
        for operation in (
            lambda: supervisor.clustering.merge([0, 1]),
            lambda: supervisor.clustering.assign([0]),
            lambda: supervisor.clustering.split([0]),
            supervisor.clustering.undo,
            supervisor.clustering.redo,
            supervisor.clustering.reset,
            lambda: supervisor.cluster_meta.set('group', [0], 'noise', add_to_stack=False),
            lambda: supervisor.cluster_meta.from_dict({0: {'group': 'noise'}}),
            lambda: controller.model.save_spike_clusters(supervisor.clustering.spike_clusters),
            lambda: controller.model.save_metadata('group', {0: 'noise'}),
            lambda: supervisor.split([0]),
        ):
            with pytest.raises(ValueError):
                operation()
        with pytest.raises(ValueError):
            supervisor.clustering.spike_clusters[0] = 8
        assert not controller.bounded_session.events
        with pytest.raises(ValueError):
            supervisor.cluster_meta.add_field('group', 'noise')
        with pytest.raises(TypeError):
            supervisor.clustering.spikes_per_cluster[0] = np.array([0])
        with pytest.raises(ValueError):
            supervisor.clustering.spikes_per_cluster[0][0] = 1
        assert 'CorrelogramView' not in controller.view_creator
        assert 'FiringRateView' not in controller.view_creator
    finally:
        controller.close()


def test_loader_and_native_history_do_not_allocate_by_sparse_unit_id(qtbot, tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    labels = np.load(params.parent / 'spike_clusters.npy').astype(np.int64)
    labels[labels == 99] = 2_000_000_000
    np.save(params.parent / 'spike_clusters.npy', labels)
    (params.parent / 'cluster_group.tsv').write_text(
        'cluster_id\tgroup\n0\tgood\n1\tmua\n2\tgood\n2000000000\tnoise\n'
    )
    output = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    controller = TemplateController(dir_path=output, config_dir=tmp_path / 'config', plugins=[])
    try:
        assert controller.model.sparse_clusters is None
        first = controller.supervisor.merge([0, 1])
        assert first.added == [2_000_000_001]
        controller.supervisor.merge([2, 2_000_000_001])
        controller.supervisor.undo()
        assert 2_000_000_001 in controller.supervisor.clustering.cluster_ids
        values = controller._window_correlograms([2, 2_000_000_001], 0.001, 0.004)
        assert values.shape == (2, 2, 5)
        controller.supervisor.save()
    finally:
        controller.close()
    reopened = TemplateController(dir_path=output, config_dir=tmp_path / 'config', plugins=[])
    try:
        assert 2_000_000_001 in reopened.model.cluster_ids
        assert reopened.model.sparse_clusters is None
    finally:
        reopened.close()


def test_merge_label_history_save_reopen_replay_regenerate(tmp_path):
    params, controller = make_controller(tmp_path)
    path = controller.dir_path
    try:
        supervisor = controller.supervisor
        original_file = (path / 'spike_clusters.npy').read_bytes()
        up = supervisor.merge([0, 1])
        assert up.added == [100]
        assert controller.get_full_count(100) == 7
        supervisor.label('group', 'noise', [100])
        supervisor.undo()
        assert controller.bounded_session.metadata['group'][100] is None
        assert supervisor.cluster_meta.get('group', 100) is None
        supervisor.redo()
        assert controller.bounded_session.metadata['group'][100] == 'noise'
        supervisor.undo()
        supervisor.undo()
        np.testing.assert_array_equal(
            supervisor.clustering.spike_clusters, controller.bounded_session.base_assignments
        )
        supervisor.merge([1, 2])
        assert supervisor.clustering.new_cluster_id() == 102
        supervisor.label('quality', 'reviewed', [101])
        supervisor.save()
        assert not supervisor.is_dirty()
        assert (path / 'spike_clusters.npy').read_bytes() == original_file
        expected = supervisor.clustering.spike_clusters.copy()
    finally:
        controller.close()
    reopened = TemplateController(
        dir_path=path, config_dir=tmp_path / 'config', plugins=[], enable_threading=False
    )
    try:
        np.testing.assert_array_equal(reopened.supervisor.clustering.spike_clusters, expected)
        assert reopened.supervisor.clustering.new_cluster_id() == 102
        assert reopened.supervisor.cluster_meta.get('quality', 101) == 'reviewed'
        assert reopened.supervisor._global_history.current_position == 0
    finally:
        reopened.close()
    output = apply_curation(params, path, tmp_path / 'curated', batch_size=2)
    source = np.load(params.parent / 'spike_clusters.npy')
    oracle = source.copy()
    oracle[np.isin(source, [1, 2])] = 101
    np.testing.assert_array_equal(np.load(output / 'spike_clusters.npy'), oracle)
    regenerated = prepare_large(
        output / 'params.py',
        tmp_path / 'regenerated',
        window_seconds=20,
        stratum_seconds=20,
        bin_seconds=5,
    )
    assert BoundedSession(regenerated).unit_ids == [0, 99, 101]


def test_source_threshold_replays_every_spike_and_full_metrics_pending(tmp_path):
    params, controller = make_controller(tmp_path)
    try:
        supervisor = controller.supervisor
        up = supervisor.split_source_amplitude(0, 4)
        assert set(up.added) == {100, 101}
        assert controller.get_full_count(100) is None
        assert controller.get_mean_firing_rate(101) is None
        supervisor.undo()
        assert controller.get_full_count(0) == 4
        supervisor.redo()
        supervisor.save()
        output = apply_curation(params, controller.dir_path, tmp_path / 'curated', batch_size=2)
        source = np.load(params.parent / 'spike_clusters.npy')
        amplitudes = np.load(params.parent / 'amplitudes.npy')
        oracle = source.copy()
        oracle[(source == 0) & (amplitudes < 4)] = 100
        oracle[(source == 0) & ~(amplitudes < 4)] = 101
        np.testing.assert_array_equal(np.load(output / 'spike_clusters.npy'), oracle)
    finally:
        controller.close()


def test_failed_save_leaves_dirty_previous_checkpoint(tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    try:
        supervisor = controller.supervisor
        supervisor.merge([0, 1])
        supervisor.save()
        supervisor.label('group', 'noise', [100])
        from phy.apps.large_recording import curation

        original = curation._write_json

        def fail(path, value):
            if path.name == 'journal.json':
                raise OSError('interrupted')
            return original(path, value)

        monkeypatch.setattr(curation, '_write_json', fail)
        with pytest.raises(OSError, match='interrupted'):
            supervisor.save()
        assert supervisor.is_dirty()
        assert BoundedSession(controller.dir_path).metadata['group'][100] is None
    finally:
        controller.close()


def test_standard_controller_retains_existing_path_and_save_format(qtbot, tmp_path):
    params = make_gui_sorting(tmp_path / 'ordinary')
    controller = TemplateController(
        dir_path=params.parent,
        config_dir=tmp_path / 'config',
        plugins=[],
        enable_threading=False,
        sample_rate=30000,
        n_channels_dat=3,
        dtype='int16',
    )
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        assert type(controller) is TemplateController
        assert controller.gui_name == 'TemplateGUI'
        assert controller.default_views[0] == 'WaveformView'
        assert 'FiringRateView' in controller.view_creator
        up = controller.supervisor.merge([0, 1])
        controller.supervisor.save()
        assert up.added == [100]
        assert not (params.parent / '.phy-curation').exists()
        assert not (params.parent / '_phy_curation_manifest.json').exists()
        assert 100 in np.load(params.parent / 'spike_clusters.npy')
    finally:
        gui.close()
        controller.close()


def test_malformed_manifest_fails_before_model_loading(tmp_path):
    params = make_sorting(tmp_path / 'ordinary')
    (params.parent / '_phy_curation_manifest.json').write_text(json.dumps({'format_version': 99}))
    with pytest.raises(ValueError, match='Unsupported'):
        TemplateController(dir_path=params.parent, plugins=[])


def test_cli_preparation_and_replay(tmp_path):
    params = make_sorting(tmp_path / 'source')
    runner = CliRunner()
    output = tmp_path / 'bounded'
    result = runner.invoke(
        phycli,
        [
            'prepare-large',
            str(params),
            str(output),
            '--window-seconds',
            '20',
            '--stratum-seconds',
            '20',
            '--bin-seconds',
            '5',
            '--batch-size',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    session = BoundedSession(output)
    session.record_merge([0, 1], 100, {'group': 'good'})
    session.save()
    result = runner.invoke(
        phycli,
        [
            'apply-curation',
            str(params),
            str(output),
            str(tmp_path / 'curated'),
            '--batch-size',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output


def test_stability_gui_and_exact_trends(qtbot, tmp_path):
    _, controller = make_controller(tmp_path, sparse=True)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    try:
        assert 'Bounded evidence' in gui.windowTitle()
        view = gui.list_views('UnitStabilityView')[0]
        data = view.get_data(0)
        assert data.counts.sum() == 4
        np.testing.assert_allclose(data.trends['amplitude'], [1.5, np.nan, 6, 9], equal_nan=True)
        view.on_select([99])  # Unit absent from evidence still has full-duration counts.
        assert view.get_data(99).counts.sum() == 2
        controller.supervisor.merge([0, 1])
        controller.supervisor.block()
        assert view.get_data(100).counts.sum() == 7
    finally:
        gui.close()
        controller.close()


def test_observer_mutation_rolls_back_native_and_semantic_state(tmp_path):
    _, controller = make_controller(tmp_path)
    supervisor = controller.supervisor
    once = []

    @connect(event='cluster', sender=supervisor.clustering)
    def mutate(sender, up):
        if not once:
            once.append(True)
            sender.assign([0])

    try:
        with pytest.raises(ValueError, match='Native assignments'):
            supervisor.merge([0, 1])
        np.testing.assert_array_equal(
            supervisor.clustering.spike_clusters, controller.bounded_session.base_assignments
        )
        assert not controller.bounded_session.events
        assert supervisor._global_history.current_position == 0
        assert not supervisor.is_dirty()
    finally:
        unconnect(mutate)
        controller.close()


def test_evidence_navigation_never_reuses_old_feature_cache(tmp_path):
    _, controller = make_controller(tmp_path, sparse=True)
    try:
        first = controller._get_features(0, channel_ids=[0, 1], load_all=True)
        assert first.spike_ids.tolist() == [0, 3]
        controller.inspect_evidence_window(1)
        assert not controller._get_feature_view_spike_ids(0, load_all=True).size
        assert not controller._get_features(0, channel_ids=[0, 1], load_all=True)
    finally:
        controller.close()


def test_late_clock_evidence_preserves_submillisecond_positions(tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    origin = 1000 * 3600 * 30000
    times = np.load(params.parent / 'spike_times.npy') + origin
    times[1:4] = origin + np.array([30, 31, 60])
    np.save(params.parent / 'spike_times.npy', times)
    params.write_text(
        params.read_text().replace('num_samples = 600000', f'num_samples = {origin + 600000}')
        + f'valid_intervals = [[{origin}, {origin + 600000}]]\n'
    )
    output = prepare_large(
        params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20, bin_seconds=3600
    )
    controller = TemplateController(dir_path=output, config_dir=tmp_path / 'config', plugins=[])
    try:
        positions = controller._window_time(np.array([1, 2]))
        assert positions[1] - positions[0] == pytest.approx(1 / 30000)
        assert np.float32(positions[0]) != np.float32(positions[1])
    finally:
        controller.close()


def test_empty_window_and_nonfinite_amplitude_measurements(qtbot, tmp_path):
    _, controller = make_controller(tmp_path, sparse=True)
    view = controller.create_evidence_amplitude_view()
    try:
        controller.inspect_evidence_window(1)
        view.on_select([0])
        assert view.data_bounds == (0, 0, view.duration, 1)
        assert all(not len(b.amplitudes) for b in view._displayed_bunchs)
    finally:
        view.canvas.close()
        controller.close()


def test_proposition_reviews_checkpoint_and_failed_review_history(tmp_path):
    from phy.cluster._propositions import MergePropositionController, decode_curation_mapping

    mapping = {'format_version': '2', 'unit_ids': [0, 1, 2, 99], 'merges': [{'unit_ids': [0, 1]}]}
    _, controller = make_controller(tmp_path, propositions=mapping)
    supervisor = controller.supervisor
    producer = controller.dir_path / 'curation.json'
    supervisor.merge_propositions = MergePropositionController(decode_curation_mapping(mapping))
    key = supervisor.merge_propositions.catalog.propositions[0].key
    try:
        with pytest.raises(RuntimeError, match='observer'), supervisor._transaction():
            supervisor.merge_propositions.accept(key, [0, 1], 100)
            raise RuntimeError('observer')
        assert supervisor.merge_propositions._history.current_position == 0
        assert not supervisor.merge_propositions.catalog.reviews
        supervisor.merge_propositions.reject(key)
        supervisor.save()
        controller.close()
        reopened = TemplateController(
            dir_path=producer.parent, config_dir=tmp_path / 'config', plugins=[]
        )
        try:
            assert (
                reopened.supervisor.merge_propositions.catalog.status_for(key).value == 'rejected'
            )
            assert not reopened.supervisor.is_dirty()
        finally:
            reopened.close()
    finally:
        controller.close()


def test_nonfinite_source_amplitudes_are_omitted_from_plots_and_kept_in_replay(qtbot, tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    amplitudes = np.load(params.parent / 'amplitudes.npy')
    amplitudes[3] = np.inf
    amplitudes[6] = np.nan
    np.save(params.parent / 'amplitudes.npy', amplitudes)
    output = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    controller = TemplateController(dir_path=output, config_dir=tmp_path / 'config', plugins=[])
    view = controller.create_evidence_amplitude_view()
    try:
        view.on_select([0])
        assert np.all(np.isfinite(view.data_bounds))
        assert all(np.all(np.isfinite(b.amplitudes)) for b in view._displayed_bunchs)
        controller.supervisor.split_source_amplitude(0, 4)
        controller.supervisor.save()
        curated = apply_curation(params, output, tmp_path / 'curated')
        labels = np.load(curated / 'spike_clusters.npy')
        assert labels[0] == 100
        assert labels[3] == labels[6] == 101
    finally:
        view.canvas.close()
        controller.close()


@pytest.mark.parametrize('operation', ['merge', 'label', 'move'])
@pytest.mark.parametrize('shortcut', [False, True])
def test_pending_navigation_settles_exact_native_and_semantic_targets(
    qtbot, tmp_path, operation, shortcut
):
    from timeit import default_timer

    params, controller = make_controller(tmp_path)
    gui = controller.create_gui(view_count={}, default_views=[], do_prompt_save=False)
    qtbot.addWidget(gui)
    supervisor = controller.supervisor
    try:
        supervisor.select([0])
        supervisor.block()
        view = supervisor.similarity_view
        view.sort_by('id', 'asc')
        view.select([1])
        supervisor.block()
        view.debouncer.delay = 60_000
        view.debouncer._last_submission_time = default_timer()
        view.next()
        view.select_toggle(1)
        view.select_toggle(1)
        assert view.get_selected_ids() == [2]
        assert supervisor.selected == [0, 1]
        assert view.debouncer.has_pending

        args = {
            'merge': (),
            'label': ('group', 'noise'),
            'move': ('noise', 'similar'),
        }[operation]
        if shortcut:
            supervisor._on_action(supervisor.action_creator, operation, *args)
        else:
            getattr(supervisor, operation)(*args)
        supervisor.block()
        event = controller.bounded_session.events[-1]
        source = np.load(params.parent / 'spike_clusters.npy')
        expected = source.copy()
        if operation == 'merge':
            assert event['parents'] == [0, 2]
            expected[np.isin(source, [0, 2])] = event['target']
            assert set(supervisor.clustering.cluster_ids) == {1, 99, event['target']}
        else:
            assert event['ids'] == ([2] if operation == 'move' else [0, 2])
            assert supervisor.cluster_meta.get('group', 2) == 'noise'
            assert supervisor.cluster_meta.get('group', 1) == 'mua'
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, expected)
        before = supervisor._global_history.current_item.selection_before
        assert before.cluster_ids == (0,)
        assert before.similar_ids == (2,)
        supervisor.save()
        replay = apply_curation(params, controller.dir_path, tmp_path / 'replay', batch_size=2)
        np.testing.assert_array_equal(np.load(replay / 'spike_clusters.npy'), expected)
        if operation != 'merge':
            replay_groups = (replay / 'cluster_group.tsv').read_text()
            assert '1\tmua\n' in replay_groups
            assert '2\tnoise\n' in replay_groups
    finally:
        gui.close()
        controller.close()
