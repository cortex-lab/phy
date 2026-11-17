"""Exact historical parent restoration, independent of retained evidence size."""

import json
from copy import deepcopy

import numpy as np
import pytest

from phy.apps.large_recording.curation import BoundedSession, interpret_events, project_assignments
from phy.apps.large_recording.live_summaries import LiveSummaryEngine, membership_expressions
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import Source
from phy.apps.large_recording.tests.test_source import make_sorting
from tools.curation_output_contract import _assignment_oracle, _history


def _package(tmp_path):
    params = make_sorting(tmp_path / 'source')
    path = prepare_large(
        params,
        tmp_path / 'bounded',
        window_seconds=20,
        stratum_seconds=20,
        batch_size=2,
        bin_seconds=5,
    )
    return params, BoundedSession(path)


def _predicate():
    return {'attribute': 'amplitudes.npy', 'operator': '<', 'threshold': 4.0, 'nonfinite': 'keep'}


def test_older_restore_preserves_later_edits_and_replay_without_threshold_split(tmp_path):
    params, session = _package(tmp_path)
    original = np.load(params.parent / 'spike_clusters.npy')
    session.record_merge([0, 1], 100)
    session.record_merge([2, 99], 101)
    session.record_label('group', [101], 'mua')
    session.record_label('notes', [100], 'merged review')
    session.record_restore_parents(100, [102, 103])
    expected = np.array([102 if v == 0 else 103 if v == 1 else 101 for v in original])
    assert session.metadata['group'] == {101: 'mua', 102: None, 103: None}
    assert session.metadata['notes'] == {102: None, 103: None}
    session.save()
    path = session.path
    session.close()
    with BoundedSession(path) as reopened:
        np.testing.assert_array_equal(reopened.assignments, expected)
        assert reopened.restored_parent_ids == {102: 0, 103: 1}
    output = apply_curation(params, path, tmp_path / 'curated', batch_size=2)
    with Source(params) as source, Source(output / 'params.py') as curated:
        np.testing.assert_array_equal(curated.arrays['spike_clusters.npy'], expected)
        assert curated.metadata['group'][101] == 'mua'
        assert curated.retired_units['100'] == {
            'operation': 'restore_parents',
            'replacements': [102, 103],
        }
        _assignment_oracle(source, curated, curated.metadata_provenance['history'], 2)
        _history(
            source,
            curated,
            json.loads((output / '_phy_curation_journal.json').read_text()),
            json.loads((path / '_phy_curation_manifest.json').read_text()),
            {101: {'group': 'mua'}},
        )
    second = prepare_large(
        output / 'params.py', tmp_path / 'second', window_seconds=20, stratum_seconds=20
    )
    with BoundedSession(second) as prepared:
        assert set(prepared.unit_ids) == {101, 102, 103}
        with pytest.raises(ValueError, match='No recoverable merge origin'):
            prepared.restore_parent_groups(101)
        prepared.record_merge([102, 103], 104)
        prepared.record_restore_parents(104, [105, 106])


def test_nested_restored_composite_survives_reopen_with_earlier_threshold(tmp_path):
    params, session = _package(tmp_path)
    original = np.load(params.parent / 'spike_clusters.npy')
    amplitudes = np.load(params.parent / 'amplitudes.npy')
    session.record_threshold_split(0, [100, 101], _predicate())
    session.record_merge([101, 1], 102)
    session.record_merge([102, 2], 103)
    session.record_label('group', [99], 'good')
    session.record_restore_parents(103, [104, 105])
    session.save()
    path = session.path
    session.close()
    with BoundedSession(path) as session:
        index, groups = session.restore_parent_groups(104)
        assert index == 1 and list(groups) == [101, 1]
        session.record_restore_parents(104, [106, 107])
        expected = original.copy().astype(np.int64)
        expected[(original == 0) & (amplitudes >= 4)] = 100
        expected[(original == 0) & (amplitudes < 4)] = 106
        expected[original == 1] = 107
        expected[original == 2] = 105
        np.testing.assert_array_equal(session.assignments, expected)
        expressions = membership_expressions(session.unit_ids, session.events)
        before = membership_expressions(session.unit_ids, session.events[:1])
        assert expressions[106] == before[101]
        assert session.metadata['group'][99] == 'good'
        session.save()
        with LiveSummaryEngine(path, batch_size=2) as engine:
            summaries = engine.resolve(expressions)
            for unit, summary in summaries.items():
                assert int(summary['counts'].sum()) == int(np.sum(expected == unit))
    output = apply_curation(params, path, tmp_path / 'curated', batch_size=2)
    with Source(params) as source, Source(output / 'params.py') as curated:
        np.testing.assert_array_equal(curated.arrays['spike_clusters.npy'], expected)
        history = curated.metadata_provenance['history']
        _assignment_oracle(source, curated, history, 1)
        _history(
            source,
            curated,
            json.loads((output / '_phy_curation_journal.json').read_text()),
            json.loads((path / '_phy_curation_manifest.json').read_text()),
            {99: {'group': 'good'}},
        )
        wrong = deepcopy(history)
        wrong[-1]['children'].reverse()
        with pytest.raises(ValueError, match='Assignment disagrees'):
            _assignment_oracle(source, curated, wrong, 2)


@pytest.mark.parametrize('mutation', ['prefix', 'revision', 'origin', 'children', 'metadata'])
def test_restore_binding_corruption_rejected_without_mutation(tmp_path, mutation):
    _, session = _package(tmp_path)
    with session:
        session.record_merge([0, 1], 100)
        event = session.preflight_restore_parents(100, [101, 102])
        if mutation == 'prefix':
            event['merge_prefix_sha256'] = '0' * 64
        elif mutation == 'revision':
            event['source_revision'] = 'other source'
        elif mutation == 'origin':
            event['merge_event_index'] = 3
        elif mutation == 'children':
            event['children'] = [101]
        else:
            event['metadata']['group'][101] = 'good'
        before = session.snapshot()
        with pytest.raises(ValueError):
            session._append(event)
        assert session.events == before['events']
        np.testing.assert_array_equal(session.assignments, before['assignments'])


def test_split_or_nested_consumed_target_is_not_restorable(tmp_path):
    _, session = _package(tmp_path)
    with session:
        session.record_merge([0, 1], 100)
        session.record_merge([100, 2], 101)
        with pytest.raises(ValueError, match='No recoverable'):
            session.preflight_restore_parents(100, [102, 103])
        session.record_threshold_split(101, [102, 103], _predicate())
        with pytest.raises(ValueError, match='No recoverable'):
            session.preflight_restore_parents(102, [104, 105])


def test_full_source_parent_absent_from_retained_evidence_is_not_omitted(tmp_path):
    from phy.apps.large_recording.tests.test_curation import make_session

    source, session = make_session(tmp_path)
    try:
        session.migrate_metadata(source.params_path)
        session.record_merge([0, 99], 100)
        _, groups = session.restore_parent_groups(100)
        assert list(groups) == [0, 99] and len(groups[99]) == 0
        session.record_restore_parents(100, [101, 102])
        assert 102 in interpret_events(session.unit_ids, session.base_metadata, session.events)[0]
        full = project_assignments(
            source.arrays['spike_clusters.npy'],
            session.events,
            {'amplitudes.npy': source.arrays['amplitudes.npy']},
        )
        assert np.sum(full == 102) == 1
    finally:
        session.close()
        source.close()


def test_native_restore_history_metadata_and_full_counts(tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller
    from phy.apps.template.gui import TemplateController

    _, controller = make_controller(tmp_path)
    path = controller.dir_path
    try:
        supervisor = controller.supervisor
        supervisor.merge([0, 1], to=100)
        supervisor.merge([2, 99], to=101)
        supervisor.label('group', 'mua', [101])
        before = controller.bounded_session.snapshot()
        out = supervisor.restore_parents(100)
        assert out.added == [102, 103]
        assert supervisor.selection.state.is_merge_mode
        assert supervisor.selected_merge == [102, 103]
        assert supervisor.parent_origins == {102: 0, 103: 1}
        assert controller.get_full_count(102) == 4
        assert controller.get_full_count(103) == 3
        assert supervisor.cluster_meta.get('group', 102) is None
        supervisor.undo()
        assert supervisor.session.events == before['events']
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before['assignments'])
        supervisor.redo()
        assert supervisor.session.events[-1]['operation'] == 'restore_parents'
        assert supervisor.parent_origins == {102: 0, 103: 1}
        supervisor.save()
    finally:
        controller.close()
    reopened = TemplateController(dir_path=path, config_dir=tmp_path / 'reopened', plugins=[])
    try:
        assert reopened.supervisor.parent_origins == {102: 0, 103: 1}
        assert reopened.get_full_count(102) == 4
        assert reopened.supervisor.cluster_meta.get('group', 101) == 'mua'
    finally:
        reopened.close()


@pytest.mark.parametrize('phase', ['append', 'capture'])
def test_post_native_restore_failure_rolls_back_all_provenance(tmp_path, monkeypatch, phase):
    from phy.apps.large_recording.tests.test_adapter import make_controller

    _, controller = make_controller(tmp_path)
    try:
        supervisor = controller.supervisor
        supervisor.merge([0, 1], to=100)
        before = supervisor.session.snapshot()
        histories = supervisor._global_history.current_position
        origins = dict(supervisor.parent_origins)
        lineage = supervisor.parent_lineage
        lineage_before = (
            (
                dict(lineage.records),
                dict(lineage.color_records),
                dict(lineage.origins),
                dict(lineage.restorations),
            )
            if lineage is not None
            else None
        )

        def fail(*args, **kwargs):
            raise RuntimeError('injected post-native restore failure')

        monkeypatch.setattr(
            supervisor.session if phase == 'append' else supervisor,
            '_append' if phase == 'append' else '_capture',
            fail,
        )
        with pytest.raises(RuntimeError, match='post-native'):
            supervisor.restore_parents(100)
        assert supervisor.session.events == before['events']
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before['assignments'])
        assert supervisor.parent_origins == origins
        assert not supervisor.clustering._restored_partitions
        assert supervisor._global_history.current_position == histories
        if lineage_before is not None:
            assert (
                lineage.records,
                lineage.color_records,
                lineage.origins,
                lineage.restorations,
            ) == lineage_before
    finally:
        controller.close()


def test_native_restore_rejects_zero_evidence_parent_before_any_mutation(tmp_path):
    from phy.apps.large_recording.supervisor import BoundedSupervisor
    from phy.apps.large_recording.tests.test_curation import make_session

    source, session = make_session(tmp_path)
    try:
        session.migrate_metadata(source.params_path)
        session.record_merge([0, 99], 100)
        supervisor = BoundedSupervisor(
            spike_clusters=session.assignments.copy(),
            session=session,
            cluster_labels=session.metadata,
        )
        before = session.snapshot()
        with pytest.raises(ValueError, match='no retained evidence'):
            supervisor.restore_parents(100)
        assert session.events == before['events']
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before['assignments'])
        assert supervisor._global_history.current_position == 0
    finally:
        session.close()
        source.close()


def test_restore_scratch_budget_subdivides_source_batches(tmp_path, monkeypatch):
    import phy.apps.large_recording.curation as curation

    params, session = _package(tmp_path)
    with session:
        session.record_merge([0, 1, 2], 100)
        session.record_restore_parents(100, [101, 102, 103])
        batches = []
        original = curation._membership_mask

        def tracked(expression, assignments, attributes, memo):
            batches.append(len(assignments))
            return original(expression, assignments, attributes, memo)

        monkeypatch.setattr(curation, 'RESTORE_SCRATCH_BYTES', 64)
        monkeypatch.setattr(curation, '_membership_mask', tracked)
        source = np.load(params.parent / 'spike_clusters.npy')
        actual = project_assignments(source, session.events)
        expected = np.array(
            [101 if v == 0 else 102 if v == 1 else 103 if v == 2 else v for v in source]
        )
        np.testing.assert_array_equal(actual, expected)
        assert batches and max(batches) == 1


def test_accepted_review_becomes_historical_across_reopen_replay_and_reprepare(tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller
    from phy.apps.template.gui import TemplateController
    from phy.cluster._propositions import PropositionStatus, proposition_key

    params, controller = make_controller(
        tmp_path,
        propositions={
            'format_version': '2',
            'unit_ids': [0, 1, 2, 99],
            'merges': [{'unit_ids': [0, 1]}],
        },
    )
    path = controller.dir_path
    key = proposition_key([0, 1])
    try:
        supervisor = controller.supervisor
        supervisor.merge_propositions.accept(key, [0, 1], 100)
        supervisor.merge([0, 1], to=100)
        reviews = supervisor.merge_propositions.catalog.review_mapping()['reviews']
        supervisor.restore_parents(100)
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        supervisor.undo()
        assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
        supervisor.redo()
        supervisor.save()
    finally:
        controller.close()

    def check(package, config):
        controller = TemplateController(dir_path=package, config_dir=config, plugins=[])
        try:
            catalog = controller.supervisor.merge_propositions.catalog
            assert catalog.status_for(key) is PropositionStatus.ACCEPTED_RESTORED
            assert catalog.review_mapping()['reviews'] == reviews
        finally:
            controller.close()

    check(path, tmp_path / 'reopened')
    output = apply_curation(params, path, tmp_path / 'curated', batch_size=2)
    second = prepare_large(
        output / 'params.py', tmp_path / 'second', window_seconds=20, stratum_seconds=20
    )
    check(second, tmp_path / 'reprepared')


def test_proposition_row_and_dock_refresh_immediately_after_restore_undo_redo(qtbot, tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller
    from phy.cluster._propositions import proposition_key

    _, controller = make_controller(
        tmp_path,
        propositions={
            'format_version': '2',
            'unit_ids': [0, 1, 2, 99],
            'merges': [{'unit_ids': [0, 1]}],
        },
    )
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        supervisor = controller.supervisor
        supervisor.block()
        key = proposition_key([0, 1])
        supervisor.merge_propositions.accept(key, [0, 1, 2], 100)
        supervisor.merge([0, 1, 2], to=100)
        supervisor.block()
        view = supervisor.merge_propositions_view
        view.select_key(key)
        reviews = supervisor.merge_propositions.catalog.review_mapping()['reviews']

        def check(status):
            selection = supervisor.selection.snapshot()
            row = view._model.row_by_id(view._id_by_key[key])
            assert row['status'] == status
            assert row['_catalog_status'] == status
            assert status in view.dock.status
            assert supervisor.merge_propositions.catalog.status_for(key).value == status
            assert supervisor.merge_propositions.catalog.review_mapping()['reviews'] == reviews
            assert supervisor.selection.snapshot() == selection

        check('accepted_modified')
        restored = supervisor.restore_parents(100)
        check('accepted_restored')
        assert 'subsequently restored' in view._model.row_by_id(view._id_by_key[key])['reason']
        assert supervisor.selected_merge == restored.added
        assert not supervisor.selection.state.merge.proposition_ids
        supervisor.undo()
        check('accepted_modified')
        supervisor.redo()
        check('accepted_restored')
        assert supervisor.selected_merge == restored.added
        assert not supervisor.selection.state.merge.proposition_ids
    finally:
        gui.close()
        controller.close()


@pytest.mark.parametrize('workers', [1, 4])
def test_recursive_restore_flattens_merges_and_stops_at_split_boundaries(tmp_path, workers):
    params, session = _package(tmp_path)
    original = np.load(params.parent / 'spike_clusters.npy')
    amplitudes = np.load(params.parent / 'amplitudes.npy')
    with session:
        session.record_threshold_split(0, [100, 101], _predicate())
        session.record_merge([101, 1], 102)
        session.record_merge([102, 2], 103)
        session.record_label('group', [99], 'good')
        index, groups = session.restore_parent_groups(103, recursive=True)
        assert index == 2 and list(groups) == [101, 1, 2]
        assert session.restore_parent_counts(103, recursive=True) == {101: None, 1: 3, 2: 3}
        event = session.preflight_restore_parents(103, [104, 105, 106], recursive=True)
        assert event['version'] == 4
        assert event['parent_references'] == [
            {'unit': 101, 'event_index': 0},
            {'unit': 1, 'event_index': -1},
            {'unit': 2, 'event_index': -1},
        ]
        session._append(event)
        expected = original.astype(np.int64).copy()
        expected[(original == 0) & (amplitudes >= 4)] = 100
        expected[(original == 0) & (amplitudes < 4)] = 104
        expected[original == 1] = 105
        expected[original == 2] = 106
        np.testing.assert_array_equal(session.assignments, expected)
        assert session.restored_parent_ids == {104: 101, 105: 1, 106: 2}
        assert session.metadata['group'][99] == 'good'
        session.save()
        path = session.path
    with BoundedSession(path) as reopened:
        np.testing.assert_array_equal(reopened.assignments, expected)
        reopened.record_merge([104, 105, 106], 107)
        assert list(reopened.restore_parent_groups(107, recursive=True)[1]) == [101, 1, 2]
        reopened.record_restore_parents(107, [108, 109, 110], recursive=True)
        reopened.save()
    output = apply_curation(params, path, tmp_path / 'curated', batch_size=2, workers=workers)
    expected[expected == 104] = 108
    expected[expected == 105] = 109
    expected[expected == 106] = 110
    with Source(params) as source, Source(output / 'params.py') as curated:
        np.testing.assert_array_equal(curated.arrays['spike_clusters.npy'], expected)
        _assignment_oracle(source, curated, curated.metadata_provenance['history'], 2)
        _history(
            source,
            curated,
            json.loads((output / '_phy_curation_journal.json').read_text()),
            json.loads((path / '_phy_curation_manifest.json').read_text()),
            {99: {'group': 'good'}},
        )


def test_v3_restores_immediate_parents_while_v4_restores_all_recorded_leaves(tmp_path):
    _, session = _package(tmp_path)
    with session:
        session.record_merge([0, 1], 100)
        session.record_merge([100, 2], 101)
        legacy = session.preflight_restore_parents(101, [102, 103])
        assert legacy['version'] == 3 and 'parent_references' not in legacy
        session._append(legacy)
        assert session.restored_parent_ids == {102: 100, 103: 2}
        assert list(session.restore_parent_groups(102, recursive=True)[1]) == [0, 1]
        session.record_merge([102, 103], 104)
        recursive = session.preflight_restore_parents(104, [105, 106, 107], recursive=True)
        before = session.snapshot()
        broken = deepcopy(recursive)
        broken['parent_references'][0]['unit'] = 99
        with pytest.raises(ValueError, match='parent references'):
            session._append(broken)
        assert session.events == before['events']
        np.testing.assert_array_equal(session.assignments, before['assignments'])
        session._append(recursive)
        assert session.restored_parent_ids == {105: 0, 106: 1, 107: 2}


def test_recursive_native_restore_is_one_undoable_action(tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller

    _, controller = make_controller(tmp_path)
    try:
        supervisor = controller.supervisor
        supervisor.merge([0, 1], to=100)
        supervisor.merge([100, 2], to=101)
        supervisor.label('group', 'mua', [99])
        before = controller.bounded_session.snapshot()
        out = supervisor.restore_parents(101)
        assert out.added == [102, 103, 104]
        assert supervisor.parent_origins == {102: 0, 103: 1, 104: 2}
        assert supervisor.selected_merge == out.added
        assert controller.bounded_session.events[-1]['version'] == 4
        assert len(controller.bounded_session.events) == len(before['events']) + 1
        supervisor.undo()
        assert controller.bounded_session.events == before['events']
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before['assignments'])
        assert supervisor.cluster_meta.get('group', 99) == 'mua'
        supervisor.redo()
        assert supervisor.selected_merge == out.added
        assert supervisor.cluster_meta.get('group', 99) == 'mua'
    finally:
        controller.close()


def test_recursive_restore_over_sixteen_leaves_and_repeated_aliases():
    from phy.apps.large_recording.source import hash_json

    units = list(range(20))
    original = np.repeat(units, 2)
    events = []
    target = 100
    events.append(
        {
            'version': 2,
            'operation': 'merge',
            'parents': [0, 1],
            'target': target,
            'metadata': {'group': None},
        }
    )
    for unit in units[2:]:
        events.append(
            {
                'version': 2,
                'operation': 'merge',
                'parents': [target, unit],
                'target': target + 1,
                'metadata': {'group': None},
            }
        )
        target += 1
    origin = len(events) - 1
    children = list(range(target + 1, target + 21))
    event = {
        'version': 4,
        'operation': 'restore_parents',
        'target': target,
        'children': children,
        'merge_event_index': origin,
        'merge_prefix_sha256': hash_json(events),
        'source_revision': 'test',
        'parent_references': [{'unit': unit, 'event_index': -1} for unit in units],
        'metadata': {'group': dict.fromkeys(children)},
    }
    events.append(event)
    active, _, _ = interpret_events(units, {}, events)
    assert active == set(children)
    np.testing.assert_array_equal(project_assignments(original, events), np.repeat(children, 2))
    broken = deepcopy(events)
    broken[-1]['parent_references'][0]['unit'] = False
    with pytest.raises(ValueError, match='parent references'):
        interpret_events(units, {}, broken)
    events.append(
        {
            'version': 2,
            'operation': 'merge',
            'parents': children,
            'target': 200,
            'metadata': {'group': None},
        }
    )
    again = deepcopy(event)
    again.update(
        target=200,
        children=list(range(201, 221)),
        merge_event_index=len(events) - 1,
        merge_prefix_sha256=hash_json(events),
        metadata={'group': dict.fromkeys(range(201, 221))},
    )
    events.append(again)
    interpret_events(units, {}, events)
    np.testing.assert_array_equal(
        project_assignments(original, events), np.repeat(range(201, 221), 2)
    )


@pytest.mark.parametrize('extra', [False, True])
def test_bounded_remerge_reassembles_review_scope_with_fresh_ids(qtbot, tmp_path, extra):
    from phy.apps.large_recording.tests.test_adapter import make_controller
    from phy.cluster._propositions import PropositionStatus, proposition_key

    _, controller = make_controller(
        tmp_path,
        propositions={
            'format_version': '2',
            'unit_ids': [0, 1, 2, 99],
            'merges': [{'unit_ids': [0, 1]}],
        },
    )
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        supervisor = controller.supervisor
        supervisor.block()
        key = proposition_key([0, 1])
        supervisor.merge_propositions.accept(key, [0, 1], 100)
        supervisor.merge([0, 1], to=100)
        original_review = supervisor.merge_propositions.catalog.review_mapping()['reviews']
        supervisor.restore_parents(100)
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        if extra:
            supervisor.add_to_merge([2])
        merged = supervisor.merge(to=104)
        assert merged is not None
        expected = PropositionStatus.ACCEPTED_MODIFIED if extra else PropositionStatus.ACCEPTED
        assert supervisor.merge_propositions.catalog.status_for(key) is expected
        assert supervisor.merge_propositions.catalog.review_mapping()['reviews'] == original_review
        supervisor.undo()
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        supervisor.redo()
        assert supervisor.merge_propositions.catalog.status_for(key) is expected
        supervisor.save()
    finally:
        gui.close()
        controller.close()
    from phy.apps.template.gui import TemplateController

    reopened = TemplateController(
        dir_path=controller.dir_path, config_dir=tmp_path / 'reopened', plugins=[]
    )
    try:
        assert reopened.supervisor.merge_propositions.catalog.status_for(key) is expected
    finally:
        reopened.close()


def test_recursive_native_restore_rejects_hidden_zero_evidence_leaf(tmp_path):
    from phy.apps.large_recording.supervisor import BoundedSupervisor
    from phy.apps.large_recording.tests.test_curation import make_session

    source, session = make_session(tmp_path)
    try:
        session.migrate_metadata(source.params_path)
        session.record_merge([0, 99], 100)
        session.record_merge([100, 1], 101)
        assert all(len(rows) for rows in session.restore_parent_groups(101)[1].values())
        supervisor = BoundedSupervisor(
            spike_clusters=session.assignments.copy(),
            session=session,
            cluster_labels=session.metadata,
        )
        before = session.snapshot()
        with pytest.raises(ValueError, match='no retained evidence'):
            supervisor.restore_parents(101)
        assert session.events == before['events']
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before['assignments'])
    finally:
        session.close()
        source.close()


def test_bounded_partial_remerge_does_not_reaccept_modified_scope(qtbot, tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller
    from phy.cluster._propositions import PropositionStatus, proposition_key

    _, controller = make_controller(
        tmp_path,
        propositions={
            'format_version': '2',
            'unit_ids': [0, 1, 2, 99],
            'merges': [{'unit_ids': [0, 1]}],
        },
    )
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        supervisor = controller.supervisor
        supervisor.block()
        key = proposition_key([0, 1])
        supervisor.merge_propositions.accept(key, [0, 1, 2], 100)
        supervisor.merge([0, 1, 2], to=100)
        restored = supervisor.restore_parents(100)
        supervisor.deselect_from_merge([restored.added[-1]])
        assert supervisor.merge(to=105) is not None
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
    finally:
        gui.close()
        controller.close()


def test_membership_containment_proves_only_structural_subset_relations():
    from phy.apps.large_recording.live_summaries import _membership_contains, _union

    source = {'kind': 'source', 'unit': 0}
    other = {'kind': 'source', 'unit': 1}
    selected = {'kind': 'split', 'parent': source, 'predicate': _predicate(), 'selected': True}
    remainder = {**selected, 'selected': False}
    all_units = _union([source, other])
    assert _membership_contains(source, selected)
    assert _membership_contains(all_units, _union([selected, other]))
    broader_filtered = {**selected, 'parent': all_units}
    assert _membership_contains(broader_filtered, selected)
    assert not _membership_contains(selected, source)
    assert not _membership_contains(remainder, selected)
    assert not _membership_contains(_union([selected, other]), all_units)
    changed_predicate = {**selected, 'predicate': {**_predicate(), 'threshold': 5.0}}
    assert not _membership_contains(changed_predicate, selected)


def test_bounded_reassembly_covers_historical_split_scope_after_reunion(qtbot, tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller
    from phy.cluster._propositions import PropositionStatus, proposition_key

    _, controller = make_controller(
        tmp_path,
        propositions={
            'format_version': '2',
            'unit_ids': [0, 1, 2, 99],
            'merges': [{'unit_ids': [1, 2]}],
        },
    )
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        supervisor = controller.supervisor
        supervisor.block()
        # Historical acceptance contains only the selected part of source unit 0.
        supervisor.split_source_amplitude(0, 4.0)
        children = controller.bounded_session.events[-1]['children']
        remainder, selected = children
        target = controller.bounded_session.next_cluster_id
        key = proposition_key([1, 2])
        supervisor.merge_propositions.accept(key, [selected, 1, 2], target)
        supervisor.merge([selected, 1, 2], to=target)
        supervisor.restore_parents(target)
        supervisor.add_to_merge([remainder])
        merged = supervisor.merge()
        assert merged is not None
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_MODIFIED
        )
        supervisor.undo()
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        supervisor.redo()
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_MODIFIED
        )
    finally:
        gui.close()
        controller.close()
