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
