"""Independent full ordinary Supervisor versus bounded journal/replay acceptance."""

import numpy as np
import pytest

import tools.compare_curation_modes as comparison
from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.tests.test_source import make_sorting


def ledger():
    return {
        'aliases': {'a': 0, 'b': 1, 'c': 99},
        'fields': ['group', 'notes'],
        'actions': [
            {'op': 'merge', 'parents': ['a', 'b'], 'as': 'ab'},
            {'op': 'undo'},
            {'op': 'redo'},
            {
                'op': 'split',
                'parent': 'ab',
                'threshold': 4.0,
                'selected': 'low',
                'remainder': 'rest',
            },
            {'op': 'label', 'ids': ['low'], 'field': 'group', 'value': 'good'},
            {'op': 'label', 'ids': ['rest'], 'field': 'group', 'value': 'noise'},
            {'op': 'label', 'ids': ['low'], 'field': 'notes', 'value': 'finite below 4'},
            {'op': 'label', 'ids': ['rest'], 'field': 'notes', 'value': 'equal/nonfinite'},
            {'op': 'undo'},
            {'op': 'redo'},
            {'op': 'save'},
            {'op': 'reopen'},
            {'op': 'save'},
        ],
        'expected_labels': {
            'low': {'group': 'good', 'notes': 'finite below 4'},
            'rest': {'group': 'noise', 'notes': 'equal/nonfinite'},
        },
    }


@pytest.mark.parametrize('budget,seed', [(5, 0), (9, 11)])
def test_ordinary_supervisor_against_bounded_all_rows(tmp_path, budget, seed):
    params = make_sorting(tmp_path / 'source')
    values = np.load(params.parent / 'amplitudes.npy')
    values[:5] = [np.nan, np.inf, -np.inf, 4, 3]
    np.save(params.parent / 'amplitudes.npy', values)
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        max_spikes=budget,
        seed=seed,
        window_seconds=3,
        stratum_seconds=20,
        batch_size=2,
    )
    ordinary = tmp_path / 'ordinary'
    comparison.run_reference(params.parent, ledger(), ordinary)
    session = BoundedSession(bounded)
    before = session.journal_state()
    merged = session.next_cluster_id
    session.record_merge([0, 1], merged)
    after = session.journal_state()
    session.restore_journal(before)
    session.restore_journal(after)
    low, rest = session.next_cluster_id, session.next_cluster_id + 1
    session.record_threshold_split(
        merged,
        [rest, low],
        {'attribute': 'amplitudes.npy', 'operator': '<', 'threshold': 4.0, 'nonfinite': 'keep'},
    )
    for alias, unit in [('low', low), ('rest', rest)]:
        for field, value in ledger()['expected_labels'][alias].items():
            session.record_label(field, [unit], value)
    session.save()
    session.close()
    reopened = BoundedSession(bounded)
    reopened.save()
    reopened.close()
    replay = tmp_path / 'replay'
    apply_curation(params, bounded, replay, batch_size=2)
    result = comparison.compare_outputs(ordinary, replay, chunk_size=2)
    assert result['ok'], result
    assert result['rows'] == len(np.load(params.parent / 'spike_clusters.npy'))
    streamed = tmp_path / 'streamed'
    report = comparison.run_reference(
        params.parent, ledger(), streamed, engine='streamed', chunk_size=2
    )
    assert report['evidence'] == 'streamed reference, NOT full ordinary GUI'
    assert comparison.compare_outputs(ordinary, streamed, chunk_size=3)['ok']


def outputs(tmp_path, left, right):
    paths = [tmp_path / 'left', tmp_path / 'right']
    for path, ids in zip(paths, (left, right), strict=True):
        path.mkdir()
        np.save(path / 'spike_clusters.npy', np.asarray(ids))
        np.save(path / 'spike_times.npy', np.arange(len(ids), dtype=np.int64))
    return paths


def test_exact_mapping_detects_conflict_across_chunks(tmp_path):
    left, right = outputs(tmp_path, [0, 1, 0, 1], [8, 9, 8, 8])
    report = comparison.compare_outputs(left, right, chunk_size=1)
    assert not report['ok']
    assert report['partition_mismatch_rows'] == 1
    assert report['examples'][0]['row'] == 3


def test_semantic_labels_follow_permuted_ids(tmp_path):
    left, right = outputs(tmp_path, [0, 1, 0], [100, 200, 100])
    comparison.write_labels(left, {'group': {0: 'good', 1: 'noise'}})
    comparison.write_labels(right, {'group': {100: 'good', 200: 'noise'}})
    assert comparison.compare_outputs(left, right)['ok']
    comparison.write_labels(right, {'group': {100: 'noise', 200: 'good'}})
    assert len(comparison.compare_outputs(left, right)['manual_label_mismatches']) == 2


@pytest.mark.parametrize('bad', ['rows', 'nan', 'times', 'shape'])
def test_invalid_outputs_fail_as_json_data(tmp_path, bad):
    left, right = outputs(tmp_path, [0, 1], [8, 9])
    if bad == 'rows':
        np.save(right / 'spike_clusters.npy', np.array([8]))
    elif bad == 'nan':
        np.save(right / 'spike_clusters.npy', np.array([8.0, np.nan]))
    elif bad == 'times':
        np.save(right / 'spike_times.npy', np.array([1, 0]))
    else:
        np.save(right / 'spike_clusters.npy', np.zeros((2, 2), dtype=int))
    result = comparison.compare_outputs(left, right)
    assert not result['ok'] and result['error']


def test_unmapped_zero_spike_label_units_require_explicit_review(tmp_path):
    left, right = outputs(tmp_path, [0, 1], [8, 9])
    comparison.write_labels(left, {'group': {0: 'good', 1: 'noise', 99: 'mua'}})
    comparison.write_labels(right, {'group': {8: 'good', 9: 'noise', 100: 'mua'}})
    result = comparison.compare_outputs(left, right)
    assert not result['ok']
    assert result['unmapped_label_ids']['left']['group'] == [99]


def test_csv_multiple_manual_fields_and_generated_info(tmp_path):
    left, right = outputs(tmp_path, [0, 1], [8, 9])
    (left / 'cluster_manual.csv').write_text('cluster_id,group,notes\n0,good,yes\n1,noise,no\n')
    (left / 'cluster_info.tsv').write_text('cluster_id\tgroup\n0\twrong\n')
    comparison.write_labels(
        right, {'group': {8: 'good', 9: 'noise'}, 'notes': {8: 'yes', 9: 'no'}}
    )
    assert comparison.compare_outputs(left, right)['ok']


def test_reference_handles_column_clusters(tmp_path):
    params = make_sorting(tmp_path / 'source', columns=True)
    comparison.run_reference(params.parent, ledger(), tmp_path / 'ordinary')
    comparison.run_reference(
        params.parent, ledger(), tmp_path / 'streamed', engine='streamed', chunk_size=2
    )
    assert comparison.compare_outputs(tmp_path / 'ordinary', tmp_path / 'streamed')['ok']


@pytest.mark.parametrize('fail', [False, True])
def test_reference_releases_supervisor_callbacks_and_states(tmp_path, monkeypatch, fail):
    import gc
    import weakref

    from phylib.utils.event import _EVENT

    from phy.cluster.supervisor import Supervisor

    params = make_sorting(tmp_path / 'source')
    before = {id(item) for item in _EVENT._callbacks}
    refs = []
    original = Supervisor.__init__

    def track(self, *args, **kwargs):
        original(self, *args, **kwargs)
        refs.append(weakref.ref(self))

    monkeypatch.setattr(Supervisor, '__init__', track)
    actions = [{'op': 'reopen'}, {'op': 'reopen'}]
    if fail:
        actions.append({'op': 'label', 'ids': ['missing'], 'field': 'group', 'value': 'good'})
    definition = {'aliases': {'a': 0}, 'fields': ['group'], 'actions': actions}
    if fail:
        with pytest.raises(KeyError):
            comparison.run_reference(params.parent, definition, tmp_path / 'reference')
    else:
        comparison.run_reference(params.parent, definition, tmp_path / 'reference')
    gc.collect()
    assert len(refs) == 3
    assert all(ref() is None for ref in refs)
    assert {id(item) for item in _EVENT._callbacks} == before


@pytest.mark.parametrize('inside', [True, False])
def test_compare_cli_cannot_overwrite_input_or_existing_report(tmp_path, monkeypatch, inside):
    import sys

    left, right = outputs(tmp_path, [0, 1], [8, 9])
    target = left / 'spike_clusters.npy' if inside else tmp_path / 'existing.json'
    if not inside:
        target.write_text('preserve this report')
    before = target.read_bytes()
    monkeypatch.setattr(
        sys,
        'argv',
        ['compare_curation_modes', 'compare', str(left), str(right), '--output', str(target)],
    )
    assert comparison.main() == 1
    assert target.read_bytes() == before


@pytest.mark.parametrize('threshold', [-100.0, 100.0])
def test_streamed_empty_split_rejected_and_snapshots_closed(tmp_path, threshold):
    params = make_sorting(tmp_path / 'source')
    definition = {
        'aliases': {'a': 0},
        'actions': [
            {
                'op': 'split',
                'parent': 'a',
                'threshold': threshold,
                'selected': 'low',
                'remainder': 'rest',
            }
        ],
    }
    with pytest.raises(ValueError, match='both selected and remainder'):
        comparison.run_reference(
            params.parent, definition, tmp_path / 'reference', engine='streamed', chunk_size=2
        )
    assert not list((tmp_path / 'reference').glob('.history-*.npy'))


def test_streamed_allocator_and_inactive_aliases_after_undo(tmp_path):
    params = make_sorting(tmp_path / 'source')
    actions = [
        {'op': 'merge', 'parents': ['a', 'b'], 'as': 'first'},
        {'op': 'undo'},
        {'op': 'merge', 'parents': ['a', 'b'], 'as': 'second'},
    ]
    definition = {'aliases': {'a': 0, 'b': 1}, 'actions': actions}
    report = comparison.run_reference(
        params.parent, definition, tmp_path / 'reference', engine='streamed', chunk_size=2
    )
    assert report['aliases']['second'] > report['aliases']['first']
    actions.append({'op': 'label', 'ids': ['first'], 'field': 'group', 'value': 'good'})
    with pytest.raises(ValueError, match='active units'):
        comparison.run_reference(
            params.parent, definition, tmp_path / 'invalid', engine='streamed', chunk_size=2
        )
