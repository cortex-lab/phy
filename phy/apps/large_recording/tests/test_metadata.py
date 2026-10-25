"""Membership-scoped metadata, source layouts and explicit legacy migration."""

import csv
import json

import numpy as np
import pytest

from phy.apps.large_recording.curation import BoundedSession, interpret_events
from phy.apps.large_recording.metadata import (
    POLICY,
    SCHEMA_FILE,
    complete_schema,
    read_metadata,
    validate_schema,
    write_metadata,
)
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import Source, hash_file
from phy.apps.large_recording.tests.test_curation import make_session
from phy.apps.large_recording.tests.test_source import make_sorting


def prepared(params, destination):
    prepare_large(
        params,
        destination,
        window_seconds=20,
        stratum_seconds=20,
        max_spikes=100,
        bin_seconds=5,
        batch_size=2,
    )
    return BoundedSession(destination)


def metadata_sorting(tmp_path):
    params = make_sorting(tmp_path / 'source')
    (params.parent / 'cluster_si_unit_ids.tsv').write_text(
        'cluster_id\tsi_unit_id\n0\t40\n1\t44\n2\t48\n88\tempty\n99\t99\n'
    )
    (params.parent / 'cluster_producer.csv').write_text(
        'unknown,cluster_id,bombcell,snr\nsua,0,good,12.26\nsua,1,good,12.92\n'
        'sua,2,mua,12.16\n,88,,\nunchanged,99,noise,3\n'
    )
    return params


def test_three_round_membership_and_noop_export(tmp_path):
    params = metadata_sorting(tmp_path)
    original = Source(params)
    expected_unchanged = {field: values.get(99) for field, values in original.metadata.items()}
    expected_empty = {field: values.get(88) for field, values in original.metadata.items()}
    original.close()
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '<',
        'threshold': 4.0,
        'nonfinite': 'keep',
    }
    previous_history = 0
    for round_id in range(3):
        session = prepared(params, tmp_path / f'bounded-{round_id}')
        assert session.metadata_policy == POLICY
        if round_id == 0:
            session.record_label('group', [0], 'good')
            session.record_merge([0, 1], 100, {'group': 'good', 'snr': 12.26})
            children = [100]
        elif round_id == 1:
            assert session.metadata['group'][100] == 'good'
            session.record_threshold_split(
                100, [101, 102], predicate, {'group': {101: 'good', 102: 'good'}}
            )
            children = [101, 102]
        else:
            session.record_merge([101, 102, 2], 103, {'group': 'good'})
            children = [103]
        for values in session.metadata.values():
            assert all(values[unit] is None for unit in children)
        for field, value in expected_unchanged.items():
            assert session.metadata[field].get(99) == value
        for field, value in expected_empty.items():
            assert session.metadata[field].get(88) == value
        session.record_label('group', children, 'good')
        session.save()
        output = apply_curation(params, session.path, tmp_path / f'round-{round_id}')
        assert (output / 'cluster_si_unit_ids.tsv').exists()
        assert not (output / 'cluster_si_unit_id.tsv').exists()
        with (output / 'cluster_producer.csv').open(newline='') as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == ['unknown', 'cluster_id', 'bombcell', 'snr']
        for unit in children:
            assert next(row for row in rows[1:] if row[1] == str(unit)) == ['', str(unit), '', '']
        current = Source(output / 'params.py')
        assert current.metadata == session.metadata
        history = current.metadata_provenance['history']
        membership = [
            record for record in history if record['operation'] in ('merge', 'threshold_split')
        ]
        latest = membership[-1]
        assert latest['parent_metadata']['group'][str(latest['parents'][0])] == 'good'
        if round_id == 0:
            assert latest['parent_metadata']['group']['1'] == 'mua'
        assert len(history) > previous_history
        previous_history = len(history)
        current.close()
        session.close()
        # A full no-op round retains the explicit label on this membership.
        noop = prepared(output / 'params.py', tmp_path / f'noop-bounded-{round_id}')
        assert all(noop.metadata['group'][unit] == 'good' for unit in children)
        noop.save()
        noop_output = apply_curation(
            output / 'params.py', noop.path, tmp_path / f'noop-output-{round_id}'
        )
        noop.close()
        params = noop_output / 'params.py'


def test_typed_tables_roundtrip_and_conflicts(tmp_path):
    schema = {
        'version': 1,
        'tables': [
            {
                'filename': 'cluster_si_unit_ids.tsv',
                'delimiter': '\t',
                'columns': ['cluster_id', 'si_unit_id'],
            },
            {
                'filename': 'cluster_metrics.csv',
                'delimiter': ',',
                'columns': ['snr', 'cluster_id', 'flag', 'details'],
            },
        ],
    }
    metadata = {
        'si_unit_id': {0: 40, 1: None},
        'snr': {0: 12.26, 1: None},
        'flag': {0: True, 1: False},
        'details': {0: ['a', 1], 1: {'x': 2}},
        'group': {0: 'good', 1: None},
    }
    write_metadata(tmp_path, metadata, schema)
    assert read_metadata(tmp_path)[0] == metadata
    assert read_metadata(tmp_path)[1] == complete_schema(schema, metadata)
    metric = tmp_path / 'cluster_metrics.csv'
    metric.write_text(metric.read_text().replace('12.26', '99.0'))
    with pytest.raises(ValueError, match='cluster_metrics.csv, field snr, unit 0'):
        read_metadata(tmp_path)


@pytest.mark.parametrize(
    'filename', ['../cluster_metric.tsv', 'cluster_info.tsv', 'cluster_bad\\evil.tsv', 'params.py']
)
def test_unsafe_or_reserved_filenames(filename):
    with pytest.raises(ValueError, match='filename'):
        validate_schema(
            {
                'version': 1,
                'tables': [
                    {'filename': filename, 'delimiter': '\t', 'columns': ['cluster_id', 'metric']}
                ],
            }
        )


def test_duplicate_fields_and_casefold_collisions(tmp_path):
    (tmp_path / 'cluster_a.tsv').write_text('cluster_id\tsnr\n0\t1\n')
    (tmp_path / 'cluster_b.csv').write_text('cluster_id,snr\n0,1\n')
    with pytest.raises(ValueError, match='Duplicate metadata field'):
        read_metadata(tmp_path)
    with pytest.raises(ValueError, match='colliding'):
        validate_schema(
            {
                'version': 1,
                'tables': [
                    {
                        'filename': 'cluster_A.tsv',
                        'delimiter': '\t',
                        'columns': ['cluster_id', 'a'],
                    },
                    {
                        'filename': 'cluster_a.tsv',
                        'delimiter': '\t',
                        'columns': ['cluster_id', 'b'],
                    },
                ],
            }
        )


def test_legacy_blank_omitted_rows_and_conflicting_edit(tmp_path):
    (tmp_path / 'cluster_metric.tsv').write_text('cluster_id\tmetric\n0\t4\n2\t\n')
    (tmp_path / '_phy_cluster_metadata.json').write_text(
        json.dumps({'metric': {'0': 4, '1': None, '2': ''}})
    )
    metadata, schema = read_metadata(tmp_path)
    assert metadata['metric'] == {0: 4, 1: None, 2: ''}
    write_metadata(tmp_path, metadata, schema)
    assert read_metadata(tmp_path)[0] == metadata
    assert '1\t\n' in (tmp_path / 'cluster_metric.tsv').read_text()
    (tmp_path / 'cluster_metric.tsv').write_text('cluster_id\tmetric\n0\t9\n1\t\n2\t\n')
    with pytest.raises(ValueError, match='field metric, unit 0'):
        read_metadata(tmp_path)


def test_legacy_migration_atomic_reopen_history_and_reviews(tmp_path):
    source, session = make_session(tmp_path)
    session.record_merge([0, 1], 100, {'group': 'good', 'producer': 9})
    session.record_label('group', [100], 'good')
    session.record_merge([100, 2], 101, {'group': 'good', 'producer': 9})
    session.record_label('comment', [101], 'explicit decision')
    review = {'opaque_review': ['unchanged']}
    legacy_generation = session.save(extra_documents={'review.json': review})
    manifest_hash = hash_file(session.path / '_phy_curation_manifest.json')
    old_hash = hash_file(legacy_generation / 'metadata.json')
    before = session.snapshot()
    event = session.migrate_metadata()
    assert event['operation'] == 'metadata_policy'
    assert session.metadata['group'][101] is None  # Parent's explicit label expires.
    assert session.metadata['producer'][101] is None
    assert session.metadata['comment'][101] == 'explicit decision'
    migrated = session.snapshot()
    session.restore(before)
    assert session.metadata['group'][101] == 'good'
    session.restore(migrated)
    assert session.metadata['group'][101] is None
    np.testing.assert_array_equal(session.assignments, before['assignments'])
    assert session.next_cluster_id == before['next_cluster_id']
    session.save()
    reopened = BoundedSession(session.path)
    assert reopened.metadata_policy == POLICY
    assert reopened.metadata == session.metadata
    assert reopened.extra_documents['review.json'] == review
    assert hash_file(session.path / '_phy_curation_manifest.json') == manifest_hash
    assert hash_file(legacy_generation / 'metadata.json') == old_hash
    reopened.record_label('group', [101], 'good')
    assert reopened.metadata['group'][101] == 'good'
    reopened.record_merge([101, 99], 102, {'group': 'good'})
    assert reopened.metadata['group'][102] is None
    source.close()
    session.close()
    reopened.close()


def test_legacy_interpretation_is_unchanged():
    events = [
        {
            'version': 1,
            'operation': 'merge',
            'parents': [0, 1],
            'target': 2,
            'metadata': {'group': 'good', 'snr': 12.26},
        }
    ]
    _, metadata, _ = interpret_events([0, 1], {'snr': {0: 12.26, 1: 8}}, events)
    assert metadata == {'group': {2: 'good'}, 'snr': {2: 12.26}}


def test_blank_legacy_metadata_preserves_revision_and_v1_values(tmp_path):
    from phy.apps.large_recording.source import assignment_revision

    params = make_sorting(tmp_path / 'full')
    (params.parent / 'cluster_metric.tsv').write_text('cluster_id\tmetric\n0\t\n')
    with Source(params) as source:
        legacy = {'group': {0: 'good', 1: 'mua', 2: 'good', 99: 'noise'}, 'metric': {0: ''}}
        assert source.metadata == legacy
        assert source.revision == assignment_revision(params.parent, legacy)
    _, projected, _ = interpret_events([0, 1], {'metric': {0: ''}}, [])
    assert projected == {'metric': {0: ''}}


def test_save_snapshot_policy_schema_and_provenance_use_supplied_state(tmp_path):
    source, session = make_session(tmp_path)
    session.record_merge([0, 1], 100, {'group': 'good'})
    legacy = session.snapshot()
    session.migrate_metadata()
    session.record_label('group', [100], 'good')
    # Save an earlier valid history prefix, as undo/history clients can do.
    session.save(**legacy)
    reopened = BoundedSession(session.path)
    assert reopened.events == legacy['events']
    assert reopened.metadata == legacy['metadata']
    assert reopened.metadata_policy is None
    assert len(reopened.metadata_provenance['history']) == 1
    source.close()
    session.close()
    reopened.close()


def test_migration_rejects_same_metadata_with_changed_source_coordinates(tmp_path):
    source, session = make_session(tmp_path)
    source.close()
    amplitudes = np.load(source.path / 'amplitudes.npy')
    amplitudes[0] += 1
    np.save(source.path / 'amplitudes.npy', amplitudes)
    with pytest.raises(ValueError, match='identity'):
        session.migrate_metadata()
    assert not session.events
    session.close()


def test_gui_membership_quality_metadata_and_history(tmp_path):
    from phy.apps.large_recording.tests.test_adapter import make_controller

    _, controller = make_controller(tmp_path)
    try:
        supervisor = controller.supervisor
        supervisor.label('producer_metric', 42, [0, 1])
        supervisor.label('classification', 'sua', [0, 1])
        supervisor.merge([0, 1])
        session = controller.bounded_session
        for field in ('producer_metric', 'classification', 'group'):
            assert session.metadata[field][100] is None
            assert supervisor.cluster_meta.get(field, 100) is None
        assert supervisor.cluster_meta.get('group', 2) == 'good'
        supervisor.label('group', 'good', [100])
        supervisor.undo()
        assert supervisor.cluster_meta.get('group', 100) is None
        supervisor.redo()
        assert supervisor.cluster_meta.get('group', 100) == 'good'
        supervisor.split_source_amplitude(100, 4)
        children = session.events[-1]['children']
        for field in ('producer_metric', 'classification', 'group'):
            assert all(supervisor.cluster_meta.get(field, child) is None for child in children)
        supervisor.undo()
        assert supervisor.cluster_meta.get('group', 100) == 'good'
        supervisor.redo()
        assert all(supervisor.cluster_meta.get('group', child) is None for child in children)
        supervisor.save()
    finally:
        controller.close()


def test_migration_keeps_explicit_target_quality_and_old_generation_reopens(tmp_path):
    source, session = make_session(tmp_path)
    # The original spelling cannot be inferred from the group header alone.
    (source.path / 'cluster_group.tsv').rename(source.path / 'cluster_quality_archive.tsv')
    session.record_merge([0, 1], 100, {'group': 'noise'})
    session.record_label('group', [100], 'good')
    session.save()
    pointer_path = session.path / '.phy-curation' / 'current.json'
    legacy_pointer = pointer_path.read_bytes()
    legacy_metadata = session.snapshot()['metadata']
    session.migrate_metadata()
    assert session.metadata['group'][100] == 'good'
    assert (
        session.effective_metadata_schema['tables'][0]['filename'] == 'cluster_quality_archive.tsv'
    )
    session.save()
    migrated_pointer = pointer_path.read_bytes()
    pointer_path.write_bytes(legacy_pointer)
    with pytest.warns(UserWarning, match='Legacy bounded metadata'):
        old = BoundedSession(session.path)
    assert old.metadata == legacy_metadata
    assert old.metadata_policy is None
    old.close()
    pointer_path.write_bytes(migrated_pointer)
    migrated = BoundedSession(session.path)
    assert migrated.metadata['group'][100] == 'good'
    assert migrated.metadata_policy == POLICY
    migrated.close()
    source.close()
    session.close()


def test_v2_resolved_metadata_cannot_smuggle_parent_measurements():
    event = {
        'version': 2,
        'operation': 'merge',
        'parents': [0, 1],
        'target': 2,
        'metadata': {'group': 'good', 'snr': 12.26},
    }
    with pytest.raises(ValueError, match='clear producer metadata'):
        interpret_events([0, 1], {'snr': {0: 12.26, 1: 8}}, [event])


def test_legacy_typed_python_table_representations(tmp_path):
    metadata = {'flag': {'0': True}, 'details': {'0': ['a', 1]}}
    (tmp_path / '_phy_cluster_metadata.json').write_text(json.dumps(metadata))
    (tmp_path / 'cluster_flag.tsv').write_text('cluster_id\tflag\n0\tTrue\n')
    (tmp_path / 'cluster_details.tsv').write_text("cluster_id\tdetails\n0\t['a', 1]\n")
    assert read_metadata(tmp_path)[0] == {'flag': {0: True}, 'details': {0: ['a', 1]}}


def test_merge_empty_catalog_unit_does_not_inherit_identity(tmp_path):
    params = metadata_sorting(tmp_path)
    session = prepared(params, tmp_path / 'bounded')
    session.record_merge([0, 88], 100)
    assert session.metadata['si_unit_id'][100] is None
    session.save()
    output = apply_curation(params, session.path, tmp_path / 'out')
    with Source(output / 'params.py') as curated:
        assert 88 not in curated.unit_ids
        assert curated.metadata['si_unit_id'][100] is None
        assert curated.counts[curated.unit_ids.tolist().index(100)] == 4
    session.close()


def test_replay_rejects_unsourced_manifest_metadata(tmp_path):
    source, session = make_session(tmp_path)
    session.close()
    manifest_path = session.path / '_phy_curation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['base_metadata']['group']['0'] = 'fabricated'
    manifest_path.write_text(json.dumps(manifest))
    session = BoundedSession(session.path)
    session.save()
    with pytest.raises(ValueError, match='base metadata does not match'):
        apply_curation(source.params_path, session.path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    source.close()
    session.close()


def test_replay_rejects_changed_original_table_basename(tmp_path):
    params = metadata_sorting(tmp_path)
    session = prepared(params, tmp_path / 'bounded')
    session.save()
    (params.parent / 'cluster_si_unit_ids.tsv').rename(params.parent / 'cluster_si_unit_id.tsv')
    with pytest.raises(ValueError, match='metadata schema changed'):
        apply_curation(params, session.path, tmp_path / 'out')
    session.close()


def test_readonly_source_symlink_preserves_table_alias_and_target(tmp_path):
    params = make_sorting(tmp_path / 'source')
    producer = tmp_path / 'immutable-producer-si.tsv'
    producer.write_text('cluster_id\tsi_unit_id\n0\t40\n1\t44\n2\t48\n99\t99\n')
    alias = params.parent / 'cluster_si_unit_ids.tsv'
    _symlink_or_skip(alias, producer)
    original_hash = hash_file(producer)
    with Source(params) as source:
        assert source.metadata['si_unit_id'][0] == '40'
        descriptor = next(
            table for table in source.metadata_schema['tables'] if 'si_unit_id' in table['columns']
        )
        assert descriptor['filename'] == 'cluster_si_unit_ids.tsv'
    session = prepared(params, tmp_path / 'bounded')
    session.save()
    output = apply_curation(params, session.path, tmp_path / 'out')
    assert (output / 'cluster_si_unit_ids.tsv').is_file()
    assert not (output / 'cluster_si_unit_ids.tsv').is_symlink()
    assert alias.is_symlink()
    assert hash_file(producer) == original_hash
    session.close()


def test_metadata_output_refuses_existing_symlink_without_mutating_target(tmp_path):
    producer = tmp_path / 'producer.tsv'
    producer.write_text('immutable producer data')
    destination = tmp_path / 'out'
    destination.mkdir()
    _symlink_or_skip(destination / 'cluster_si_unit_ids.tsv', producer)
    schema = {
        'version': 1,
        'tables': [
            {
                'filename': 'cluster_si_unit_ids.tsv',
                'delimiter': '\t',
                'columns': ['cluster_id', 'si_unit_id'],
            }
        ],
    }
    original_hash = hash_file(producer)
    with pytest.raises(ValueError, match='output must not overwrite a symbolic link'):
        write_metadata(destination, {'si_unit_id': {0: '40'}}, schema)
    assert hash_file(producer) == original_hash


def _symlink_or_skip(path, target):
    try:
        path.symlink_to(target)
    except OSError as error:
        if getattr(error, 'winerror', None) == 1314:
            pytest.skip('This Windows account has no symbolic-link privilege.')
        raise
