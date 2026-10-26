"""Reject undeclared full-output changes, including internally consistent metadata edits."""

import csv
import hashlib
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import tools.compare_curation_modes as comparison
from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.metadata import SCHEMA_FILE, read_metadata, write_metadata
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import assignment_revision, hash_file
from phy.apps.large_recording.tests.test_source import make_sorting


def _inventory(path):
    return {
        str(item.relative_to(path)): hash_file(item) for item in path.rglob('*') if item.is_file()
    }


def _pair(tmp_path, aliases=False, twin=False, split=False, unclassified=False):
    params = make_sorting(tmp_path / 'source')
    source = params.parent
    assert sum((twin, split, unclassified)) <= 1
    if unclassified:
        (source / 'cluster_group.tsv').unlink()
    params.write_text(params.read_text() + "exporter_tag = 'fixture-v1'\n")
    (source / 'cluster_snr.tsv').write_text(
        'cluster_id\tsnr\n0\t12\n1\t12\n2\t12\n99\t12\n'
        if twin
        else 'cluster_id\tsnr\n0\t12\n1\t13\n2\t22\n99\t3\n'
    )
    (source / 'cluster_notes.tsv').write_text(
        'cluster_id\tnotes\n0\tfirst\n1\tsecond\n2\tcontrol\n99\tlast\n'
    )
    (source / 'cluster_empty.tsv').write_text('cluster_id\tempty\n0\t\n1\t\n2\t\n99\t\n')
    np.save(source / '_phy_spikes_subset.window_bounds.npy', np.array([[0, 20]], dtype=np.int64))
    (source / 'curation.json').write_text(
        '{"format_version":"2","unit_ids":[0,1,2,99],"merges":[{"unit_ids":[0,1]}]}\n'
    )
    if aliases:
        (source / 'cluster_si_unit_ids.tsv').write_text(
            'cluster_id\tsi_unit_id\n0\t10\n1\t11\n2\t12\n99\t99\n'
        )
        (source / 'cluster_import.csv').write_text(
            'producer,cluster_id,score\nfirst,0,1.5\nsecond,1,2.5\ncontrol,2,3.5\nlast,99,4.5\n'
        )
    original = _inventory(source)
    fields = list(read_metadata(source)[0])
    if 'group' not in fields:
        fields.append('group')
    definition = {
        'aliases': {'a': 0, 'b': 1, 'c': 2, 'd': 99},
        'fields': fields,
        'actions': [
            {'op': 'merge', 'parents': ['a', 'b'], 'as': 'merged'},
            {'op': 'label', 'ids': ['merged'], 'field': 'group', 'value': 'good'},
            {'op': 'label', 'ids': ['merged'], 'field': 'notes', 'value': 'reviewed'},
            {'op': 'save'},
        ],
        'expected_labels': {'merged': {'group': 'good', 'notes': 'reviewed'}},
    }
    if split:
        definition['actions'] = [
            {'op': 'merge', 'parents': ['a', 'b'], 'as': 'merged'},
            {
                'op': 'split',
                'parent': 'merged',
                'threshold': 4.0,
                'selected': 'low',
                'remainder': 'rest',
            },
            *[
                {'op': 'label', 'ids': [alias], 'field': field, 'value': value}
                for alias in ('low', 'rest')
                for field, value in (('group', 'good'), ('notes', 'reviewed'))
            ],
            {'op': 'save'},
        ]
        definition['expected_labels'] = {
            alias: {'group': 'good', 'notes': 'reviewed'} for alias in ('low', 'rest')
        }
    if unclassified:
        definition['actions'] = [
            a for a in definition['actions'] if not (a['op'] == 'label' and a['field'] == 'group')
        ]
        definition['expected_labels'] = {'merged': {'notes': 'reviewed'}}
    if twin:
        definition['actions'][-1:-1] = [
            {'op': 'merge', 'parents': ['c', 'd'], 'as': 'twin'},
            {'op': 'label', 'ids': ['twin'], 'field': 'group', 'value': 'good'},
            {'op': 'label', 'ids': ['twin'], 'field': 'notes', 'value': 'reviewed'},
        ]
    ordinary = tmp_path / 'ordinary-full'
    shutil.copytree(source, ordinary)
    reference = tmp_path / 'ordinary-reference'
    comparison.run_reference(source, definition, reference)
    for item in reference.iterdir():
        if item.name == 'spike_clusters.npy' or item.name.startswith('cluster_'):
            shutil.copyfile(item, ordinary / item.name)
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        window_seconds=20,
        stratum_seconds=20,
        max_spikes=100,
        bin_seconds=5,
        batch_size=2,
    )
    session = BoundedSession(bounded)
    try:
        merged = session.next_cluster_id
        assert merged == 100
        session.record_merge([0, 1], merged)
        targets = [merged]
        if split:
            session.record_threshold_split(
                merged,
                [102, 101],
                {
                    'attribute': 'amplitudes.npy',
                    'operator': '<',
                    'threshold': 4.0,
                    'nonfinite': 'keep',
                },
            )
            targets = [101, 102]
        for unit in targets:
            if not unclassified:
                session.record_label('group', [unit], 'good')
            session.record_label('notes', [unit], 'reviewed')
        if twin:
            session.record_merge([2, 99], 101)
            session.record_label('group', [101], 'good')
            session.record_label('notes', [101], 'reviewed')
        session.save()
    finally:
        session.close()
    right = apply_curation(params, bounded, tmp_path / 'replay', batch_size=2)
    return SimpleNamespace(
        source=source,
        ordinary=ordinary,
        right=right,
        bounded=bounded,
        expected={
            unit: (
                {'notes': 'reviewed'} if unclassified else {'group': 'good', 'notes': 'reviewed'}
            )
            for unit in ([101, 102] if split else [100, 101] if twin else [100])
        },
        original=original,
    )


def _compare(pair, bounded=True):
    return comparison.compare_outputs(
        pair.ordinary,
        pair.right,
        fields=('group', 'notes'),
        chunk_size=2,
        source=pair.source,
        expected_labels=pair.expected,
        bounded_dir=pair.bounded if bounded else None,
    )


@pytest.fixture
def contract_pair(tmp_path):
    pair = _pair(tmp_path)
    try:
        yield pair
    finally:
        assert _inventory(pair.source) == pair.original


def _change_json(path, mutate):
    content = json.loads(path.read_text())
    mutate(content)
    path.write_text(json.dumps(content) + '\n')


def _refresh_revision(right):
    metadata, _ = read_metadata(right)
    _change_json(
        right / '_phy_curation_lineage.json',
        lambda document: document.update(revision=assignment_revision(right, metadata)),
    )


def _edit_metadata(right, field, unit, value):
    metadata, schema = read_metadata(right)
    metadata[field][unit] = value
    write_metadata(right, metadata, schema)
    # Keep the typed sidecar, every TSV and the lineage revision consistent.
    # The checker must reject the metadata policy violation, not just a stale hash.
    _refresh_revision(right)


def _journal_hash(journal):
    payload = json.dumps(journal, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@pytest.mark.parametrize('aliases', [False, True])
def test_complete_file_contract_accepts_declared_metadata_differences(tmp_path, aliases):
    pair = _pair(tmp_path, aliases=aliases)
    try:
        result = _compare(pair)
        assert result['ok'], result
        assert result['rows'] == 12
        metadata, _ = read_metadata(pair.right)
        assert metadata['snr'][100] is None
        assert metadata['snr'][2] == '22'
        assert metadata['group'][100] == 'good'
        if aliases:
            for name in ('cluster_si_unit_ids.tsv', 'cluster_import.csv'):
                assert (pair.ordinary / name).read_bytes() == (pair.source / name).read_bytes()
            assert (pair.ordinary / 'cluster_si_unit_id.tsv').exists()
            assert (pair.ordinary / 'cluster_producer.tsv').exists()
    finally:
        assert _inventory(pair.source) == pair.original


@pytest.mark.parametrize('name', ['cluster_si_unit_ids.tsv', 'cluster_import.csv'])
def test_ordinary_source_alias_allowance_requires_unchanged_bytes(tmp_path, name):
    pair = _pair(tmp_path, aliases=True)
    try:
        assert _compare(pair)['ok']
        path = pair.ordinary / name
        # Change only a retired parent's value in the loose producer alias.
        # Active canonical fields and the curation outputs still agree.
        if name.endswith('.tsv'):
            path.write_text(path.read_text().replace('0\t10', '0\t777'))
        else:
            path.write_text(path.read_text().replace('first,0,1.5', 'forged,0,1.5'))
        assert not _compare(pair)['ok']
    finally:
        assert _inventory(pair.source) == pair.original


def test_first_round_source_binding_checks_journal_without_bounded_package(contract_pair):
    pair = contract_pair
    assert _compare(pair, bounded=False)['ok']
    path = pair.right / '_phy_curation_journal.json'
    journal = json.loads(path.read_text())
    journal[-1]['value'] = 'forged label intent'
    path.write_text(json.dumps(journal) + '\n')
    _change_json(
        pair.right / '_phy_curation_lineage.json',
        lambda document: document.update(journal_sha256=_journal_hash(journal)),
    )
    result = _compare(pair, bounded=False)
    assert not result['ok']
    assert 'Journal details disagree with cumulative history' in result['error']


def test_ordinary_derived_whitening_inverse_must_be_correct(contract_pair):
    pair = contract_pair
    assert not (pair.source / 'whitening_mat.npy').exists()
    assert not (pair.source / 'whitening_mat_inv.npy').exists()
    inverse = np.eye(3, dtype=np.float64)
    path = pair.ordinary / 'whitening_mat_inv.npy'
    np.save(path, inverse)
    assert _compare(pair)['ok']
    inverse[0, 0] = 2
    np.save(path, inverse)
    result = _compare(pair)
    assert not result['ok']
    assert 'Invalid native derived whitening inverse' in result['error']


@pytest.mark.parametrize(
    'corruption',
    [
        'amplitude',
        'waveform',
        'channels',
        'window_bounds',
        'assignment',
        'manual_label',
        'changed_membership_producer',
        'untouched_producer',
        'ordinary_untouched_producer',
        'clock',
        'exposure',
        'auxiliary_parameter',
        'new_parameter',
        'missing_asset',
        'unknown_file',
        'ordinary_unknown_file',
        'metadata_filename',
        'metadata_header',
        'metadata_schema',
        'journal',
        'parent_revision',
        'review_binding',
        'ordinary_missing_empty_table',
        'unrecorded_empty_field',
        'malformed_history',
        'parent_metadata_history',
    ],
)
def test_complete_file_contract_rejects_output_corruption(contract_pair, corruption):
    pair = contract_pair
    right = pair.right
    assert _compare(pair)['ok']
    assets = {
        'amplitude': ('amplitudes.npy', (0,)),
        'waveform': ('_phy_spikes_subset.waveforms.npy', (0, 0, 0)),
        'channels': ('channel_positions.npy', (0, 0)),
        'window_bounds': ('_phy_spikes_subset.window_bounds.npy', (0, 1)),
    }
    if corruption in assets:
        name, index = assets[corruption]
        values = np.load(right / name)
        values[index] += 1
        np.save(right / name, values)
    elif corruption == 'assignment':
        values = np.load(right / 'spike_clusters.npy')
        values[0] = 2
        np.save(right / 'spike_clusters.npy', values)
        _refresh_revision(right)
    elif corruption == 'manual_label':
        _edit_metadata(right, 'group', 100, 'noise')
    elif corruption == 'changed_membership_producer':
        _edit_metadata(right, 'snr', 100, '999')
    elif corruption == 'untouched_producer':
        _edit_metadata(right, 'snr', 2, '999')
    elif corruption == 'ordinary_untouched_producer':
        path = pair.ordinary / 'cluster_snr.tsv'
        path.write_text(path.read_text().replace('2\t22', '2\t999'))
    elif corruption in ('clock', 'exposure', 'auxiliary_parameter', 'new_parameter'):
        changes = {
            'clock': 'sample_rate = 2.0\n',
            'exposure': 'valid_intervals = [[0, 3], [5, 20]]\n',
            'auxiliary_parameter': "exporter_tag = 'changed'\n",
            'new_parameter': "unexplained_option = 'changed'\n",
        }
        path = right / 'params.py'
        path.write_text(path.read_text() + changes[corruption])
    elif corruption == 'missing_asset':
        (right / 'spike_templates.npy').unlink()
    elif corruption == 'unknown_file':
        (right / 'unexplained.txt').write_text('This is not an allowed generated output.\n')
    elif corruption == 'ordinary_unknown_file':
        (pair.ordinary / 'unexplained.txt').write_text('Not an ordinary GUI cache or table.\n')
    elif corruption == 'metadata_filename':
        (right / 'cluster_snr.tsv').rename(right / 'cluster_snrs.tsv')

        def rename(document):
            for table in document['tables']:
                if table['filename'] == 'cluster_snr.tsv':
                    table['filename'] = 'cluster_snrs.tsv'

        _change_json(right / SCHEMA_FILE, rename)
    elif corruption == 'metadata_header':
        path = right / 'cluster_snr.tsv'
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream, delimiter='\t'))
        with path.open('w', newline='') as stream:
            csv.writer(stream, delimiter='\t').writerows([list(reversed(row)) for row in rows])

        def reorder(document):
            for table in document['tables']:
                if table['filename'] == 'cluster_snr.tsv':
                    table['columns'].reverse()

        _change_json(right / SCHEMA_FILE, reorder)
    elif corruption == 'metadata_schema':
        _change_json(right / SCHEMA_FILE, lambda document: document.update(version=2))
    elif corruption == 'journal':
        path = right / '_phy_curation_journal.json'
        journal = json.loads(path.read_text())
        journal[0]['target'] = 101
        path.write_text(json.dumps(journal) + '\n')
        _change_json(
            right / '_phy_curation_lineage.json',
            lambda document: document.update(journal_sha256=_journal_hash(journal)),
        )
    elif corruption == 'parent_revision':
        _change_json(
            right / '_phy_curation_lineage.json',
            lambda document: document.update(parent_revision='0' * 64),
        )
    elif corruption == 'review_binding':
        path = right / 'curation_review.json'
        _change_json(path, lambda document: document['source'].update(sha256='0' * 64))
        _change_json(
            right / '_phy_curation_lineage.json',
            lambda document: document['proposition_documents'].update(
                {'curation_review.json': hash_file(path)}
            ),
        )
    elif corruption == 'ordinary_missing_empty_table':
        (pair.ordinary / 'cluster_empty.tsv').unlink()
    elif corruption == 'unrecorded_empty_field':
        metadata, schema = read_metadata(right)
        metadata['invented'] = dict.fromkeys(metadata['group'])
        write_metadata(right, metadata, schema)
        _refresh_revision(right)
    elif corruption == 'malformed_history':
        _change_json(
            right / '_phy_curation_lineage.json',
            lambda document: document.update(metadata_provenance=[]),
        )
    elif corruption == 'parent_metadata_history':

        def change_snapshot(document):
            document['metadata_provenance']['history'][0]['parent_metadata']['snr']['0'] = '999'

        _change_json(right / '_phy_curation_lineage.json', change_snapshot)
    else:
        raise AssertionError(corruption)
    result = _compare(pair)
    assert not result['ok'], (corruption, result)
    if corruption == 'journal':
        assert 'Journal details disagree with cumulative history' in result['error']
    elif corruption == 'ordinary_missing_empty_table':
        assert 'Missing native metadata field/table coverage' in result['error']
    elif corruption == 'unrecorded_empty_field':
        assert 'Unexpected replay metadata fields without label intent' in result['error']
    elif corruption == 'parent_metadata_history':
        assert 'Parent metadata provenance disagrees with independent history' in result['error']
    elif corruption == 'malformed_history':
        assert result['error']


def test_strict_contract_standalone_cli(contract_pair, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    pair = contract_pair
    repository = Path(__file__).resolve().parents[4]
    expectations = tmp_path / 'expectations.json'
    expectations.write_text(json.dumps(pair.expected))
    output = tmp_path / 'cli-report.json'
    result = subprocess.run(
        [
            sys.executable,
            str(repository / 'tools/compare_curation_modes.py'),
            'compare',
            str(pair.ordinary),
            str(pair.right),
            '--source',
            str(pair.source),
            '--bounded-dir',
            str(pair.bounded),
            '--expectations',
            str(expectations),
            '--output',
            str(output),
            '--chunk-size',
            '2',
        ],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report['ok'] and report['file_contract']['saved_generation_verified']


def test_source_only_contract_checks_new_unit_row_ancestry(tmp_path):
    pair = _pair(tmp_path, twin=True)
    assert _compare(pair)['ok']
    values = np.load(pair.right / 'spike_clusters.npy')
    swapped = np.where(values == 100, 101, 100).astype(values.dtype)
    np.save(pair.right / 'spike_clusters.npy', swapped)
    _refresh_revision(pair.right)
    # Same partitions and identical labels allow this rename in the basic check;
    # the stored provenance still names the WRONG source parents for those IDs.
    assert comparison.compare_outputs(pair.ordinary, pair.right)['ok']
    result = comparison.compare_outputs(
        pair.ordinary,
        pair.right,
        source=pair.source,
        expected_labels=pair.expected,
        chunk_size=2,
    )
    assert not result['ok'] and 'ancestry' in result['error']
    assert _inventory(pair.source) == pair.original


@pytest.mark.parametrize('corruption', ['allocator', 'retirement'])
def test_saved_generation_and_retired_identity_binding(contract_pair, corruption):
    pair = contract_pair
    path = pair.right / '_phy_curation_lineage.json'
    if corruption == 'allocator':
        _change_json(path, lambda d: d.update(next_cluster_id=d['next_cluster_id'] + 1))
    else:
        _change_json(
            path,
            lambda d: d['retired_units'].update(
                {'999': {'operation': 'merge', 'replacements': [100]}}
            ),
        )
    result = _compare(pair)
    assert not result['ok']
    assert ('saved checkpoint' if corruption == 'allocator' else 'independent history') in result[
        'error'
    ]


def test_generated_summary_requires_complete_active_catalog(contract_pair):
    pair = contract_pair
    path = pair.ordinary / 'cluster_info.tsv'
    path.write_text(
        'cluster_id\tgroup\tnotes\n2\tgood\tcontrol\n99\tnoise\tlast\n100\tgood\treviewed\n'
    )
    assert _compare(pair)['ok']
    path.write_text('cluster_id\tgroup\tnotes\n99\tnoise\tlast\n100\tgood\treviewed\n')
    result = _compare(pair)
    assert not result['ok'] and 'cluster_info catalog' in result['error']


def test_native_blank_source_column_can_retain_retired_rows(contract_pair):
    pair = contract_pair
    path = pair.ordinary / 'cluster_empty.tsv'
    shutil.copyfile(pair.source / path.name, path)
    result = _compare(pair)
    assert result['ok'], result
    assert any(
        a['policy'] == 'byte-pinned blank source column'
        for a in result['file_contract']['ordinary_aliases']
    )
    path.write_text(path.read_text().replace('0\t\n', '0\tforged\n'))
    assert not _compare(pair)['ok']


def test_membership_can_introduce_implicit_unreviewed_group(tmp_path):
    pair = _pair(tmp_path, unclassified=True)
    result = _compare(pair)
    assert result['ok'], result.get('error', result)
    assert _inventory(pair.source) == pair.original


@pytest.mark.parametrize('corruption', ['metadata', 'extra_payload', 'single_parent'])
def test_source_only_rejects_invalid_membership_journal(contract_pair, corruption):
    pair = contract_pair
    journal_path = pair.right / '_phy_curation_journal.json'
    lineage_path = pair.right / '_phy_curation_lineage.json'
    journal = json.loads(journal_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    if corruption == 'metadata':
        journal[0]['metadata']['snr'] = 'forged producer metadata'
    elif corruption == 'extra_payload':
        journal[0]['unexplained'] = True
    else:
        journal[0]['parents'] = [0]
        lineage['metadata_provenance']['history'][0]['parents'] = [0]
    journal_path.write_text(json.dumps(journal) + '\n')
    lineage['journal_sha256'] = _journal_hash(journal)
    lineage_path.write_text(json.dumps(lineage) + '\n')
    report = comparison.compare_outputs(
        pair.ordinary,
        pair.right,
        source=pair.source,
        expected_labels=pair.expected,
        chunk_size=2,
    )
    assert not report['ok']
    expected = {
        'metadata': 'clearing policy',
        'extra_payload': 'event schema',
        'single_parent': 'parent cardinality',
    }
    assert expected[corruption] in report['error'], report


@pytest.mark.parametrize(
    'key,value',
    [
        ('operator', '=='),
        ('attribute', 'spike_depths.npy'),
        ('threshold', True),
        ('nonfinite', 'drop'),
    ],
)
def test_source_only_rejects_invalid_threshold_predicate(tmp_path, key, value):
    pair = _pair(tmp_path, split=True)
    assert _compare(pair)['ok']
    path = pair.right / '_phy_curation_journal.json'
    journal = json.loads(path.read_text())
    journal[1]['predicate'][key] = value
    path.write_text(json.dumps(journal) + '\n')
    lineage_path = pair.right / '_phy_curation_lineage.json'
    lineage = json.loads(lineage_path.read_text())
    lineage['journal_sha256'] = _journal_hash(journal)
    lineage['metadata_provenance']['history'][1]['predicate'][key] = value
    lineage_path.write_text(json.dumps(lineage) + '\n')
    result = comparison.compare_outputs(
        pair.ordinary,
        pair.right,
        source=pair.source,
        expected_labels=pair.expected,
        chunk_size=2,
    )
    assert not result['ok'] and 'predicate schema' in result['error'], result
    assert _inventory(pair.source) == pair.original


def test_native_transformed_producer_difference_is_ancestry_bounded(contract_pair):
    pair = contract_pair
    path = pair.ordinary / 'cluster_snr.tsv'
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream, delimiter='\t'))
    rows = [row for row in rows if row[0] != '100']
    rows.append(['100', '12'])  # A real source parent's metric, not a new measurement.
    with path.open('w', newline='') as stream:
        csv.writer(stream, delimiter='\t').writerows(rows)
    result = _compare(pair)
    assert result['ok'], result
    assert any(
        d['field'] == 'snr' and d['right_id'] == 100 and d['left'] == '12' and d['right'] == ''
        for d in result['permitted_membership_policy_differences']
    )
    rows[-1] = ['100', '999']
    with path.open('w', newline='') as stream:
        csv.writer(stream, delimiter='\t').writerows(rows)
    result = _compare(pair)
    assert not result['ok'] and 'unsupported ancestor value' in result['error']
