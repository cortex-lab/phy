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
