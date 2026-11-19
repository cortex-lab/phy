"""Compiled projection agrees with the independent sequential interpreter."""

import numpy as np
import pytest

from phy.apps.large_recording.curation import project_assignments
from phy.apps.large_recording.replay_program import AssignmentProgram
from phy.apps.large_recording.tests.test_restore_parents import _package, _predicate


def test_many_merges_compile_to_one_map_with_sparse_ids():
    units = [0, 2, 1000000, 2000000000]
    original = np.array(units * 5, dtype=np.int64)
    events = [
        {'operation': 'merge', 'parents': [0, 2], 'target': 2000000001},
        {'operation': 'label', 'field': 'notes', 'ids': [1000000], 'value': 'x'},
        {'operation': 'merge', 'parents': [1000000, 2000000001], 'target': 2000000002},
    ]
    program = AssignmentProgram(units, events)
    assert program.merge_runs == 1
    np.testing.assert_array_equal(program.project(original), project_assignments(original, events))
    events[0]['parents'].append(2000000000)
    np.testing.assert_array_equal(program.project(original)[3::4], 2000000000)


@pytest.mark.parametrize('recursive', [False, True])
@pytest.mark.parametrize('batch_size', [1, 3, 100])
def test_mixed_split_nested_restore_and_merge_matches_reference(tmp_path, recursive, batch_size):
    params, session = _package(tmp_path)
    try:
        session.record_threshold_split(0, [100, 101], _predicate())
        session.record_merge([101, 1], 102)
        session.record_merge([102, 2], 103)
        session.record_label('notes', [103], 'inspect')
        children = [104, 105, 106] if recursive else [104, 105]
        session.record_restore_parents(103, children, recursive=recursive)
        session.record_merge([100, children[0]], 107)
        original = np.load(params.parent / 'spike_clusters.npy')
        attributes = {'amplitudes.npy': np.load(params.parent / 'amplitudes.npy')}
        expected = project_assignments(original, session.events, attributes)
        program = AssignmentProgram(session.unit_ids, session.events)
        actual = np.concatenate(
            [
                program.project(
                    original[start : start + batch_size],
                    {
                        name: values[start : start + batch_size]
                        for name, values in attributes.items()
                    },
                )
                for start in range(0, len(original), batch_size)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
    finally:
        session.close()


@pytest.mark.parametrize('policy', ['keep', 'reject'])
@pytest.mark.parametrize('operator', ['<', '<=', '>', '>='])
def test_predicate_boundaries_and_rejection_survive_remerging(policy, operator):
    original = np.array([0, 0, 0, 1, 1])
    values = np.array([0.5, 1, np.nan, np.inf, 2])
    events = [
        {'operation': 'merge', 'parents': [0, 1], 'target': 2},
        {
            'operation': 'threshold_split',
            'parent': 2,
            'children': [3, 4],
            'predicate': {
                'attribute': 'amplitudes.npy',
                'operator': operator,
                'threshold': 1,
                'nonfinite': policy,
            },
        },
        {'operation': 'merge', 'parents': [3, 4], 'target': 5},
    ]
    program = AssignmentProgram([0, 1], events)
    if policy == 'reject':
        with pytest.raises(ValueError, match='Nonfinite'):
            program.project(original, {'amplitudes.npy': values})
    else:
        np.testing.assert_array_equal(
            program.project(original, {'amplitudes.npy': values}),
            project_assignments(original, events, {'amplitudes.npy': values}),
        )
    with pytest.raises(ValueError, match='unavailable'):
        program.project(original)
    with pytest.raises(ValueError, match='scalar'):
        program.project(original, {'amplitudes.npy': values[:-1]})


def test_empty_batch_and_uint64_precision():
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '>',
        'threshold': 2**63 + 1,
        'nonfinite': 'keep',
    }
    events = [
        {'operation': 'threshold_split', 'parent': 0, 'children': [1, 2], 'predicate': predicate}
    ]
    program = AssignmentProgram([0], events)
    values = np.array([2**63, 2**63 + 1, 2**63 + 2], dtype=np.uint64)
    original = np.zeros(3, dtype=np.int64)
    np.testing.assert_array_equal(program.project(original, {'amplitudes.npy': values}), [1, 1, 2])
    assert program.project(original[:0], {'amplitudes.npy': values[:0]}).size == 0


def test_compiled_writer_corruption_is_rejected_by_independent_verifier(tmp_path, monkeypatch):
    from phy.apps.large_recording.replay import apply_curation

    params, session = _package(tmp_path)
    try:
        session.record_threshold_split(0, [100, 101], _predicate())
        session.record_merge([101, 1], 102)
        session.save()
        original = AssignmentProgram.project

        def corrupted(self, assignments, attributes=None):
            return np.roll(original(self, assignments, attributes), 1)

        monkeypatch.setattr(AssignmentProgram, 'project', corrupted)
        output = tmp_path / 'corrupt-output'
        with pytest.raises(ValueError, match='semantic replay verification'):
            apply_curation(params, session.path, output, batch_size=100)
        assert not output.exists()
        assert not list(tmp_path.glob('.corrupt-output-*'))
    finally:
        session.close()
