"""Interactive commands must not replay unrelated historical spike assignments."""

import numpy as np
import pytest

from phy.apps.large_recording.tests.test_curation import make_session


def test_label_has_no_assignment_projection(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    source, session = make_session(tmp_path)
    try:
        session.record_merge([0, 1], 100, {'group': 'good'})
        assignments = session.assignments

        def forbidden(*args, **kwargs):
            pytest.fail('A label projected spike assignments.')

        monkeypatch.setattr(curation, 'project_assignments', forbidden)
        event = session.preflight_label('group', [100], 'mua')
        assert session.events[-1]['operation'] == 'merge'
        session._append(event)
        assert session.assignments is assignments
        assert session.metadata['group'][100] == 'mua'
        with pytest.raises(ValueError, match='inactive'):
            session.record_label('group', [0], 'good')
        assert session.assignments is assignments
    finally:
        session.close()
        source.close()


def test_incremental_events_and_history_match_full_oracle(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation
    from phy.apps.large_recording.tests.test_restore_parents import _package, _predicate

    _, session = _package(tmp_path)
    oracle = curation.project_assignments
    interpreter = curation.interpret_events
    states = []

    def one_event(units, metadata, events):
        assert len(events) <= 1, 'An interactive command reinterpreted its journal.'
        return interpreter(units, metadata, events)

    def forbidden(*args, **kwargs):
        pytest.fail('An interactive command replayed its journal.')

    try:
        monkeypatch.setattr(curation, 'project_assignments', forbidden)
        monkeypatch.setattr(curation, 'interpret_events', one_event)
        states.append(session.journal_state())
        commands = [
            lambda: session.record_merge([0, 1], 100),
            lambda: session.record_threshold_split(100, [101, 102], _predicate()),
            lambda: session.record_merge([102, 2], 103),
            lambda: session.record_label('notes', [103], {'checked': True}),
            lambda: session.record_restore_parents(103, [104, 105], recursive=True),
        ]
        assignments = session.assignments
        for command in commands:
            command()
            assert session.assignments is assignments
            states.append(session.journal_state())
            expected = oracle(session.base_assignments, session.events, session.attributes)
            np.testing.assert_array_equal(session.assignments, expected)
            active, metadata, _ = interpreter(
                session.unit_ids, session.base_metadata, session.events
            )
            assert set(session._runtime.active) == active
            assert session.metadata == metadata
        for state in [*reversed(states), *states]:
            session.restore_journal(state)
            expected = oracle(session.base_assignments, session.events, session.attributes)
            np.testing.assert_array_equal(session.assignments, expected)
            assert session.assignments is assignments
        session.restore_journal(states[2])
        token = session.membership_revision
        session.record_label('group', [101], 'good')
        assert session.membership_revision is token
        assert session.next_cluster_id == 106
        session.record_merge([101, 102], 106)
        assert session.membership_revision is not token
        assert session.next_cluster_id == 107
        expected = oracle(session.base_assignments, session.events, session.attributes)
        np.testing.assert_array_equal(session.assignments, expected)
    finally:
        session.close()


def test_history_cache_eviction_replays_and_preserves_allocator(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    source, session = make_session(tmp_path)
    try:
        session.history_cache_bytes = 0
        before = session.journal_state()
        session.record_merge([0, 1], 100, {'group': 'good'})
        after = session.journal_state()
        assert not session._transitions and session._transition_bytes == 0
        calls = []
        original = curation.project_assignments

        def measured(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(curation, 'project_assignments', measured)
        session.restore_journal(before)
        assert calls and session.next_cluster_id == 101
        np.testing.assert_array_equal(session.assignments, session.base_assignments)
        session.restore_journal(after)
        assert len(calls) == 2
        assert session.metadata['group'][100] == 'good'
    finally:
        session.close()
        source.close()


def test_prepared_event_revalidates_allocator_and_detaches_values(tmp_path):
    source, session = make_session(tmp_path)
    try:
        event = session.preflight_merge([0, 1], 100)
        session.next_cluster_id = 200
        with pytest.raises(ValueError, match='allocator'):
            session._append(event)
        assert not session.events
        event = session.preflight_label('notes', [0], {'checked': []})
        session._append(event)
        event['value']['checked'].append('caller mutation')
        assert session.events[-1]['value'] == {'checked': []}
        assert session.metadata['notes'][0] == {'checked': []}
    finally:
        session.close()
        source.close()


def test_metadata_only_journal_has_constant_interpretation_scope(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation, runtime

    source, session = make_session(tmp_path)
    scopes = []
    interpreter = curation.interpret_events

    def measured(units, metadata, events):
        scopes.append((len(units), len(events)))
        return interpreter(units, metadata, events)

    try:
        monkeypatch.setattr(runtime, 'interpret_events', measured)
        assignments = session.assignments
        revision = session.membership_revision
        for i in range(1000):
            session.record_label('group', [0], 'good' if i % 2 else 'mua')
        assert set(scopes) == {(4, 1)}
        assert session.assignments is assignments
        assert session.membership_revision is revision
        assert all(not t.old_groups and not t.new_groups for t in session._transitions.values())
    finally:
        session.close()
        source.close()


def test_shared_journal_records_are_immutable_and_snapshots_detached(tmp_path):
    source, session = make_session(tmp_path)
    try:
        session.record_merge([0, 1], 100)
        state = session.journal_state()
        with pytest.raises(TypeError, match='read-only'):
            state['events'][0]['parents'][0] = 123456
        with pytest.raises(TypeError, match='read-only'):
            session.events.append({'operation': 'bad'})
        with pytest.raises(AttributeError):
            session.events = []
        snapshot = session.snapshot()
        snapshot['events'][0]['parents'][0] = 123456
        with pytest.raises(ValueError, match='inactive'):
            session.restore(snapshot)
        assert session.events[0]['parents'] == [0, 1]
        session.restore_journal(state)
        assert session.metadata == session._runtime.metadata
    finally:
        session.close()
        source.close()


def test_restore_plan_expressions_are_detached_from_runtime(tmp_path):
    from phy.apps.large_recording.tests.test_restore_parents import _package

    _, session = _package(tmp_path)
    try:
        session.record_merge([0, 1], 100)
        expressions = session.restore_parent_plan(100)[2]
        expressions[0]['unit'] = 123456
        assert session.restore_parent_plan(100)[2][0]['unit'] == 0
        session.record_restore_parents(100, [101, 102], recursive=True)
        assert set(session.assignments) == {2, 99, 101, 102}
    finally:
        session.close()


def test_save_preserves_cached_prefix_and_explicit_snapshot_rebuilds_runtime(
    tmp_path, monkeypatch
):
    from phy.apps.large_recording import curation

    source, session = make_session(tmp_path)
    original = curation.project_assignments
    try:
        baseline = session.snapshot()
        before = session.journal_state()
        session.record_merge([0, 1], 100)
        after = session.journal_state()
        event = session.events[-1]
        session.save()
        assert session.events[-1] is event

        def forbid(*args, **kwargs):
            pytest.fail('Saving discarded cached incremental history.')

        monkeypatch.setattr(curation, 'project_assignments', forbid)
        session.restore_journal(before)
        session.restore_journal(after)
        monkeypatch.setattr(curation, 'project_assignments', original)
        session.save(**baseline)
        assert set(session._runtime.active) == set(session.unit_ids)
        assert not session.events
        session.record_label('group', [0], 'mua')
        session.record_merge([0, 2], 100)
        expected = original(session.base_assignments, session.events, session.attributes)
        np.testing.assert_array_equal(session.assignments, expected)
    finally:
        session.close()
        source.close()


def test_history_budget_accounts_for_nested_metadata(tmp_path):
    source, session = make_session(tmp_path)
    try:
        session.history_cache_bytes = 64 * 1024
        session.record_label('notes', [0], {'values': [0] * 100000})
        for _ in range(3):
            session.record_label('group', [0], 'good')
        assert not session._transitions
        assert session._transition_bytes == 0
        assert session.metadata['notes'][0]['values'] == [0] * 100000
    finally:
        session.close()
        source.close()


def test_cyclic_label_payload_is_rejected_before_freezing(tmp_path):
    source, session = make_session(tmp_path)
    try:
        cyclic = []
        cyclic.append(cyclic)
        with pytest.raises(ValueError, match='Circular reference'):
            session.record_label('notes', [0], cyclic)
        assert not session.events
    finally:
        session.close()
        source.close()


@pytest.mark.parametrize('budget', [0, 1])
def test_evicted_transition_releases_previous_row_arrays(tmp_path, budget):
    import gc
    import weakref

    source, session = make_session(tmp_path)
    try:
        session.history_cache_bytes = budget
        old_rows = weakref.ref(session._assignment_groups[0])
        event = session.preflight_merge([0, 1], 100)
        transition = weakref.ref(session._prepared)
        session._append(event)
        gc.collect()
        assert not session._transitions
        assert session._transition_bytes == 0
        assert transition() is None
        assert old_rows() is None
        assert not hasattr(session, 'last_transition')
    finally:
        session.close()
        source.close()
