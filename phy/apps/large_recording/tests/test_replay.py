"""Full assignment oracle, revision binding and atomic publication regressions."""

import json

import numpy as np
import pytest

from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import Source, hash_file
from phy.apps.large_recording.tests.test_curation import make_session


def test_full_replay_chains_preserve_unseen_units_and_immutable_bytes(tmp_path):
    source, session = make_session(tmp_path)
    original_hashes = {name: hash_file(source.path / name) for name in source.arrays}
    original = source.arrays['spike_clusters.npy'].copy()
    session.record_merge([0, 1], 100, {'group': 'good', 'confidence': 0.8})
    session.record_label('group', [100], 'mua')
    session.record_merge([100, 2], 101, {'group': 'good', 'confidence': None})
    session.record_label('custom', [99, 101], 'verified')
    session.save()
    output = apply_curation(source.params_path, session.path, tmp_path / 'curated', batch_size=3)
    expected = np.where(np.isin(original, [0, 1, 2]), 101, original)
    assert np.array_equal(np.load(output / 'spike_clusters.npy'), expected)
    assert expected[-1] == 99
    curated = Source(output / 'params.py')
    assert curated.metadata == session.metadata
    assert curated.identity == source.identity
    assert curated.revision != source.revision
    lineage = json.loads((output / '_phy_curation_lineage.json').read_text())
    assert lineage['parent_revision'] == source.revision
    assert lineage['revision'] == curated.revision
    for name, digest in original_hashes.items():
        assert hash_file(source.path / name) == digest
        if name != 'spike_clusters.npy':
            assert hash_file(output / name) == digest
    with pytest.raises(ValueError, match='revision'):
        apply_curation(output / 'params.py', session.path, tmp_path / 'twice')
    assert not (tmp_path / 'twice').exists()
    second = apply_curation(source.params_path, session.path, tmp_path / 'second', batch_size=7)
    assert hash_file(second / 'spike_clusters.npy') == hash_file(output / 'spike_clusters.npy')


def test_source_revision_and_feature_identity_are_independently_bound(tmp_path):
    source, session = make_session(tmp_path)
    session.save()
    clusters = source.arrays['spike_clusters.npy'].copy()
    clusters[0] = 2
    np.save(source.path / 'spike_clusters.npy', clusters)
    with pytest.raises(ValueError, match='revision'):
        apply_curation(source.params_path, session.path, tmp_path / 'wrong_revision')
    amplitudes = source.arrays['amplitudes.npy'].copy()
    amplitudes[0] += 1
    np.save(source.path / 'amplitudes.npy', amplitudes)
    with pytest.raises(ValueError, match='identity'):
        apply_curation(source.params_path, session.path, tmp_path / 'wrong_feature')
    assert not (tmp_path / 'wrong_revision').exists()
    assert not (tmp_path / 'wrong_feature').exists()


def test_full_threshold_split_matches_scalar_oracle(tmp_path):
    source, session = make_session(tmp_path)
    original = source.arrays['spike_clusters.npy'].copy()
    amplitudes = source.arrays['amplitudes.npy'].copy()
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '<',
        'threshold': 15.0,
        'nonfinite': 'keep',
    }
    session.record_split(2, 100, 101, predicate, {'group': {100: 'noise', 101: 'good'}})
    session.record_merge([0, 100], 102, {'group': 'mua'})
    session.save()
    expected = original.astype(np.int64)
    expected[(original == 2) & (amplitudes < 15)] = 100
    expected[(original == 2) & ~(amplitudes < 15)] = 101
    expected[np.isin(expected, [0, 100])] = 102
    output = apply_curation(source.params_path, session.path, tmp_path / 'split', batch_size=4)
    assert np.array_equal(np.load(output / 'spike_clusters.npy'), expected)
    assert Source(output / 'params.py').metadata == session.metadata


def test_failed_replay_does_not_publish_output(tmp_path, monkeypatch):
    from phy.apps.large_recording import replay

    source, session = make_session(tmp_path)
    session.save()
    original = replay._write_json

    def fail(path, value):
        if path.name == '_phy_curation_lineage.json':
            raise OSError('injected replay interruption')
        return original(path, value)

    monkeypatch.setattr(replay, '_write_json', fail)
    with pytest.raises(OSError, match='interruption'):
        apply_curation(source.params_path, session.path, tmp_path / 'interrupted')
    assert not (tmp_path / 'interrupted').exists()
    assert not list(tmp_path.glob('.interrupted-*'))


def test_replay_requires_committed_checkpoint_and_new_output(tmp_path):
    source, session = make_session(tmp_path)
    with pytest.raises(ValueError, match='Save'):
        apply_curation(source.params_path, session.path, tmp_path / 'unsaved')
    session.save()
    with pytest.raises(FileExistsError):
        apply_curation(source.params_path, session.path, source.path)


def test_replay_preserves_empty_split_children_and_allocator_lineage(tmp_path):
    source, session = make_session(tmp_path)
    session.record_split(
        2,
        100,
        101,
        {
            'attribute': 'amplitudes.npy',
            'operator': '<',
            'threshold': 100.0,
            'nonfinite': 'keep',
        },
    )
    session.save()
    output = apply_curation(source.params_path, session.path, tmp_path / 'empty_child')
    curated = Source(output / 'params.py')
    assert 101 in curated.unit_ids
    assert curated.counts[curated.unit_ids.tolist().index(101)] == 0
    assert curated.next_cluster_id == 102


def test_bounded_threshold_attribute_must_match_exact_source_rows(tmp_path):
    source, session = make_session(tmp_path)
    session.save()
    # Even a manifest that does not hash this optional attribute must not allow
    # evidence values different from those used by the full-data predicate.
    values = np.load(session.path / 'amplitudes.npy')
    values[0] += 1
    np.save(session.path / 'amplitudes.npy', values)
    with pytest.raises(ValueError, match='does not match source rows'):
        apply_curation(source.params_path, session.path, tmp_path / 'bad_evidence')
    assert not (tmp_path / 'bad_evidence').exists()
