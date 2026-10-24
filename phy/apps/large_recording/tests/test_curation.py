"""Exact bounded journal projection and transactional persistence regressions."""

import json
import shutil

import numpy as np
import pytest

from phy.apps.large_recording.curation import BASE_ASSIGNMENTS, BoundedSession
from phy.apps.large_recording.source import Source, hash_file
from phy.apps.tests.large_recording_fixtures import make_long_recording_fixture


def make_session(tmp_path):
    fixture = make_long_recording_fixture(tmp_path)
    # The highest complete-source unit is intentionally absent from evidence.
    clusters = np.load(fixture.full / 'spike_clusters.npy')
    clusters[-1] = 99
    np.save(fixture.full / 'spike_clusters.npy', clusters)
    (fixture.full / 'cluster_group.tsv').write_text('cluster_id\tgroup\n0\tgood\n1\tnoise\n')
    source = Source(fixture.full / 'params.py', batch_size=5)
    rows = np.array([0, 2, 5, 8, 13, 21, 25], dtype=np.int64)
    for name in ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy', 'amplitudes.npy'):
        np.save(fixture.bounded / name, source.arrays[name][rows])
    np.save(fixture.bounded / BASE_ASSIGNMENTS, clusters[rows])
    np.save(fixture.bounded / '_phy_spike_global_ids.npy', rows)
    np.save(fixture.bounded / 'windows.npy', np.array([[0, source.num_samples]], dtype=np.int64))
    manifest = {
        'format': 'phy_bounded_curation',
        'format_version': 1,
        'source': {
            'identity': source.identity,
            'revision': source.revision,
            'path': str(source.params_path),
        },
        'sample_rate': source.sample_rate,
        'num_samples': source.num_samples,
        'valid_intervals': source.valid_intervals.tolist(),
        'num_spikes': source.n_spikes,
        'bounded_num_spikes': len(rows),
        'unit_ids': source.unit_ids.tolist(),
        'counts': source.counts.tolist(),
        'next_cluster_id': 100,
        'global_spike_ids_file': '_phy_spike_global_ids.npy',
        'windows_file': 'windows.npy',
        'base_metadata': source.metadata,
        'capabilities': {
            'merge': True,
            'label': True,
            'split': False,
            'predicate_split_attributes': ['amplitudes.npy'],
        },
        'bounded_files': {
            name: hash_file(fixture.bounded / name)
            for name in (
                BASE_ASSIGNMENTS,
                '_phy_spike_global_ids.npy',
                'windows.npy',
                'spike_times.npy',
            )
        },
    }
    (fixture.bounded / '_phy_curation_manifest.json').write_text(json.dumps(manifest))
    return source, BoundedSession(fixture.bounded)


@pytest.mark.parametrize('command', ['record_threshold_split', 'record_split'])
def test_split_numpy_metadata_keys_save_and_reopen(tmp_path, command):
    source, session = make_session(tmp_path)
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '<',
        'threshold': 0.5,
        'nonfinite': 'keep',
    }
    children = [np.int64(100), np.int64(101)]
    metadata = {'group': {children[0]: 'good', children[1]: 'noise'}}
    try:
        if command == 'record_threshold_split':
            session.record_threshold_split(0, children, predicate, metadata)
        else:
            session.record_split(0, *children, predicate, metadata)
        session.save()
        with_metadata = session.events[-1]['metadata']['group']
        assert all(type(key) is int for key in with_metadata)
        reopened = BoundedSession(session.path)
        try:
            np.testing.assert_array_equal(reopened.assignments, session.assignments)
            assert reopened.metadata['group'][100] == 'good'
            assert reopened.metadata['group'][101] == 'noise'
        finally:
            reopened.close()
    finally:
        session.close()
        source.close()


def test_merge_label_snapshot_branch_reopen(tmp_path):
    source, session = make_session(tmp_path)
    baseline = session.snapshot()
    session.record_merge([0, 1], 100, {'group': 'good'})
    first = session.snapshot()
    session.record_label('group', [100], 'mua')
    second = session.snapshot()
    session.restore(first)
    assert session.metadata['group'][100] == 'good'
    session.restore(second)
    assert session.metadata['group'][100] == 'mua'
    session.restore(baseline)
    session.record_merge([1, 2], 100, {'group': 'noise'})
    session.record_label('custom', [99, 100], 'reviewed')
    session.save(extra_documents={'curation_review.json': {'accepted': [100]}})
    reopened = BoundedSession(session.path)
    assert reopened.next_cluster_id == 101
    assert reopened.events == session.events
    assert reopened.metadata == session.metadata
    assert np.array_equal(reopened.assignments, session.assignments)
    assert reopened.extra_documents == {'curation_review.json': {'accepted': [100]}}
    assert source.unit_ids.tolist() == [0, 1, 2, 99]
    assert 99 not in session.base_assignments


@pytest.mark.parametrize('parents,target', [([0, 1], 99), ([0, 0], 100), ([0, 3], 100)])
def test_merge_collision_and_invalid_parents_do_not_mutate(tmp_path, parents, target):
    _, session = make_session(tmp_path)
    before = session.snapshot()
    with pytest.raises(ValueError):
        session.record_merge(parents, target)
    assert session.events == before['events']
    assert np.array_equal(session.assignments, before['assignments'])


def test_save_requires_exact_semantic_projection(tmp_path):
    _, session = make_session(tmp_path)
    session.record_merge([0, 1], 100, {'group': 'good'})
    with pytest.raises(ValueError, match='projection'):
        session.save(assignments=session.base_assignments)
    with pytest.raises(ValueError, match='metadata'):
        session.save(metadata={})
    assert not (session.path / '.phy-curation' / 'current.json').exists()


@pytest.mark.parametrize('fail_at', ['journal.json', 'checkpoint.json', 'pointer'])
def test_interrupted_save_recovers_previous_generation(tmp_path, monkeypatch, fail_at):
    from phy.apps.large_recording import curation

    _, session = make_session(tmp_path)
    session.record_merge([0, 1], 100, {'group': 'good'})
    session.save()
    previous = session.snapshot()
    session.record_label('group', [100], 'mua')
    original = curation._write_json

    def fail(path, value):
        if path.name == fail_at or (fail_at == 'pointer' and path.name.startswith('.current-')):
            raise OSError('injected interrupted write')
        return original(path, value)

    monkeypatch.setattr(curation, '_write_json', fail)
    with pytest.raises(OSError, match='interrupted'):
        session.save()
    reopened = BoundedSession(session.path)
    assert reopened.events == previous['events']
    assert reopened.metadata == previous['metadata']
    assert session.metadata['group'][100] == 'mua'


def test_checkpoint_corruption_is_rejected(tmp_path):
    _, session = make_session(tmp_path)
    session.save()
    (session.checkpoint_path / 'journal.json').write_text('[]\nchanged')
    with pytest.raises(ValueError, match='corruption'):
        BoundedSession(session.path)


def test_bound_base_corruption_is_rejected(tmp_path):
    _, session = make_session(tmp_path)
    session.close()
    shutil.copyfile(session.path / 'spike_times.npy', session.path / BASE_ASSIGNMENTS)
    with pytest.raises(ValueError, match='integrity'):
        BoundedSession(session.path)


def test_threshold_projection_is_serialized_and_nonfinite_policy_explicit(tmp_path):
    _, session = make_session(tmp_path)
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '<',
        'threshold': 15.0,
        'nonfinite': 'keep',
    }
    session.record_split(2, 100, 101, predicate, {'group': {100: 'noise', 101: 'good'}})
    expected = session.base_assignments.astype(np.int64).copy()
    mask = expected == 2
    selected = np.load(session.path / 'amplitudes.npy') < 15
    expected[mask & selected] = 100
    expected[mask & ~selected] = 101
    assert np.array_equal(session.assignments, expected)
    session.save()
    assert BoundedSession(session.path).events == session.events
    with pytest.raises(ValueError, match='Unsupported'):
        session.record_split(0, 102, 103, dict(predicate, attribute='waveform_peak'))


@pytest.mark.parametrize('boundary', ['generation', 'pointer'])
def test_atomic_publication_failure_preserves_previous_commit(tmp_path, monkeypatch, boundary):
    from phy.apps.large_recording import curation

    _, session = make_session(tmp_path)
    session.save()
    session.record_merge([0, 1], 100, {'group': 'good'})
    original = curation.os.replace

    def fail(source, destination):
        if (boundary == 'generation' and destination.parent.name == 'generations') or (
            boundary == 'pointer' and destination.name == 'current.json'
        ):
            raise OSError('injected atomic publication failure')
        return original(source, destination)

    monkeypatch.setattr(curation.os, 'replace', fail)
    with pytest.raises(OSError, match='publication'):
        session.save()
    assert BoundedSession(session.path).events == []
    assert len(session.events) == 1


def test_nonfinite_predicate_keep_and_reject_are_explicit():
    from phy.apps.large_recording.curation import evaluate_predicate

    values = np.array([1.0, np.nan, np.inf, -np.inf, 3.0])
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '<',
        'threshold': 2.0,
        'nonfinite': 'keep',
    }
    assert evaluate_predicate(values, predicate).tolist() == [True, False, False, False, False]
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate_predicate(values, dict(predicate, nonfinite='reject'))


def test_semantically_corrupt_checkpoint_rejected_even_with_valid_file_hashes(tmp_path):
    from phy.apps.large_recording.curation import _digest

    _, session = make_session(tmp_path)
    session.record_merge([0, 1], 100, {'group': 'good'})
    session.save()
    generation = session.checkpoint_path
    journal = json.loads((generation / 'journal.json').read_text())
    journal[0]['target'] = 99
    (generation / 'journal.json').write_text(json.dumps(journal))
    descriptor = json.loads((generation / 'checkpoint.json').read_text())
    descriptor['files']['journal.json'] = hash_file(generation / 'journal.json')
    (generation / 'checkpoint.json').write_text(json.dumps(descriptor))
    pointer = {'generation': generation.name, 'sha256': _digest(descriptor)}
    (session.path / '.phy-curation' / 'current.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='colliding'):
        BoundedSession(session.path)


def test_compact_history_preserves_allocator_and_reconstructs_assignments(tmp_path):
    _, session = make_session(tmp_path)
    before = session.journal_state()
    session.record_merge([0, 1], 100, {'group': 'good'})
    after = session.journal_state()
    merged = session.assignments.copy()
    assert set(after) == {'events', 'next_cluster_id'}
    assert after['events'][0] is session.events[0]
    session.restore_journal(before)
    assert session.events == []
    assert session.next_cluster_id == 101
    assert np.array_equal(session.assignments, session.base_assignments)
    session.restore_journal(after)
    assert np.array_equal(session.assignments, merged)
    session.restore_journal(before)
    session.record_merge([1, 2], 101, {'group': 'noise'})
    assert session.next_cluster_id == 102


def test_session_close_releases_own_file_mappings(tmp_path):
    _, session = make_session(tmp_path)
    mappings = [
        array._mmap
        for array in (
            session.global_ids,
            session.base_assignments,
            *session.attributes.values(),
        )
    ]
    session.close()
    assert all(mapping.closed for mapping in mappings)
    session.close()


def test_stale_session_save_rejected_without_losing_first_commit(tmp_path):
    _, first = make_session(tmp_path)
    second = BoundedSession(first.path)
    first.record_merge([0, 1], 100, {'group': 'good'})
    second.record_merge([1, 2], 100, {'group': 'noise'})
    first.save(extra_documents={'curation_review.json': {'accepted': [100]}})
    with pytest.raises(ValueError, match='reopen'):
        second.save(extra_documents={'curation_review.json': {'rejected': [100]}})
    reopened = BoundedSession(first.path)
    assert reopened.events == first.events
    assert reopened.extra_documents == first.extra_documents
    assert second.events != reopened.events
    # The winning session can commit again without reopening.
    first.record_label('group', [100], 'mua')
    first.save()
    assert BoundedSession(first.path).metadata['group'][100] == 'mua'


def test_failed_save_releases_commit_lock(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    _, session = make_session(tmp_path)
    session.save()
    other = BoundedSession(session.path)
    original = curation._write_json

    def fail(path, value):
        if path.name == 'journal.json':
            raise OSError('injected locked save failure')
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(curation, '_write_json', fail)
        with pytest.raises(OSError, match='locked'):
            session.save()
    other.record_label('group', [0], 'mua')
    other.save()
    assert BoundedSession(session.path).metadata['group'][0] == 'mua'


def _concurrent_save_worker(path, barrier, results, label):
    session = BoundedSession(path)
    try:
        session.record_label('group', [0], label)
        barrier.wait(timeout=20)
        try:
            session.save()
        except ValueError as error:
            results.put(('conflict', str(error)))
        else:
            results.put(('saved', label))
    finally:
        session.close()


def test_concurrent_process_saves_compare_generation_under_lock(tmp_path):
    import multiprocessing

    _, session = make_session(tmp_path)
    session.save()
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_save_worker,
            args=(str(session.path), barrier, results, label),
        )
        for label in ('good', 'noise')
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert sorted(status for status, _ in outcomes) == ['conflict', 'saved']
        winner = next(value for status, value in outcomes if status == 'saved')
        assert 'reopen' in next(value for status, value in outcomes if status == 'conflict')
        assert BoundedSession(session.path).metadata['group'][0] == winner
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()


def _replace_capabilities(session, capabilities):
    session.close()
    path = session.path / '_phy_curation_manifest.json'
    manifest = json.loads(path.read_text())
    if capabilities is None:
        manifest.pop('capabilities', None)
    else:
        manifest['capabilities'] = capabilities
    path.write_text(json.dumps(manifest))
    return BoundedSession(session.path)


@pytest.mark.parametrize(
    'capabilities', [None, {}, {'merge': False, 'label': False, 'split': False}]
)
def test_readonly_capabilities_reject_public_mutations_without_state_changes(
    tmp_path, capabilities
):
    _, session = make_session(tmp_path)
    session = _replace_capabilities(session, capabilities)
    before = session.snapshot()
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '<',
        'threshold': 15.0,
        'nonfinite': 'keep',
    }
    commands = [
        lambda: session.preflight_merge([0, 1], 100),
        lambda: session.record_merge([0, 1], 100),
        lambda: session.preflight_label('group', [0], 'good'),
        lambda: session.record_label('group', [0], 'good'),
        lambda: session.preflight_split(2, 100, 101, predicate),
        lambda: session.record_split(2, 100, 101, predicate),
        lambda: session.record_threshold_split(2, [101, 100], predicate),
    ]
    for command in commands:
        with pytest.raises(ValueError, match='capability'):
            command()
        assert session.events == before['events']
        assert session.metadata == before['metadata']
        assert session.next_cluster_id == before['next_cluster_id']
        assert np.array_equal(session.assignments, before['assignments'])


@pytest.mark.parametrize(
    'capabilities',
    [
        {'merge': 'true'},
        {'label': 'false'},
        {'stored_waveforms': 1},
        {'raw_traces': None},
        {'split': True},
        {'predicate_split_attributes': 'amplitudes.npy'},
        {'predicate_split_attributes': ['amplitudes.npy', 'amplitudes.npy']},
        {'predicate_split_attributes': ['spike_templates.npy']},
        {'predicate_split_attributes': ['waveform_peak']},
        {'predicate_split_attributes': [True]},
        {'full_attributes': 'amplitude'},
    ],
)
def test_malformed_or_unsupported_capabilities_fail_validation(tmp_path, capabilities):
    _, session = make_session(tmp_path)
    with pytest.raises(ValueError, match='capability|capabilities|unsupported|Unsupported'):
        _replace_capabilities(session, capabilities)


def test_predicate_capability_requires_an_available_scalar_source_attribute(tmp_path):
    _, session = make_session(tmp_path)
    session.close()
    values = np.load(session.path / 'amplitudes.npy')
    np.save(session.path / 'amplitudes.npy', np.column_stack([values, values]))
    with pytest.raises(ValueError, match='source scalar'):
        BoundedSession(session.path)
