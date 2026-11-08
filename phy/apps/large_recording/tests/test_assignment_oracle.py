"""Literal all-source expectations for the independent acceptance oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.curation_output_contract import _assignment_oracle


def _split(
    parent, children, operator='<', threshold=4, nonfinite='keep', attribute='amplitudes.npy'
):
    return {
        'operation': 'threshold_split',
        'parents': [parent],
        'children': children,
        'predicate': {
            'attribute': attribute,
            'operator': operator,
            'threshold': threshold,
            'nonfinite': nonfinite,
        },
    }


def _check(original, values, expected, history, chunk_size=2, attribute='amplitudes.npy'):
    source = SimpleNamespace(
        arrays={
            'spike_clusters.npy': np.asarray(original, dtype=np.int16),
            attribute: np.asarray(values),
        }
    )
    curated = SimpleNamespace(arrays={'spike_clusters.npy': np.asarray(expected)})
    return _assignment_oracle(source, curated, history, chunk_size)


@pytest.mark.parametrize('attribute', ['amplitudes.npy', 'spike_templates.npy'])
@pytest.mark.parametrize('column', [False, True])
@pytest.mark.parametrize(
    ('operator', 'expected'),
    [
        ('<', [40001, 40000, 40000, 9]),
        ('<=', [40001, 40001, 40000, 9]),
        ('>', [40000, 40000, 40001, 9]),
        ('>=', [40000, 40001, 40001, 9]),
    ],
)
def test_predicate_boundaries_and_child_order(attribute, column, operator, expected):
    values = np.array([1, 4, 7, 2])
    if column:
        values = values[:, None]
    report = _check(
        [0, 0, 0, 9],
        values,
        expected,
        [_split(0, [40000, 40001], operator=operator, attribute=attribute)],
        attribute=attribute,
    )
    assert report == {
        'verified': True,
        'source_rows': 4,
        'membership_events': 1,
        'threshold_splits': 1,
    }


@pytest.mark.parametrize('operator', ['<', '<=', '>', '>='])
def test_nonfinite_keep_stays_in_remainder(operator):
    _check(
        [0, 0, 0, 9],
        [np.nan, np.inf, -np.inf, np.nan],
        [10, 10, 10, 9],
        [_split(0, [10, 11], operator=operator)],
    )


def test_nonfinite_reject_only_applies_to_current_parent():
    history = [_split(0, [10, 11], nonfinite='reject')]
    _check([0, 0, 9], [1, 7, np.nan], [11, 10, 9], history)
    with pytest.raises(ValueError, match='rejects nonfinite'):
        _check([0, 0, 9], [np.nan, 7, 1], [10, 10, 9], history)


@pytest.mark.parametrize('chunk_size', [1, 2, 7, 100])
def test_cumulative_merge_split_remerge_and_split(chunk_size):
    history = [
        {'operation': 'merge', 'parents': [0, 1], 'children': [10]},
        _split(10, [11, 12]),
        {'operation': 'merge', 'parents': [12, 2], 'children': [13]},
        {'operation': 'label', 'field': 'group', 'ids': [13], 'value': 'good'},
        _split(13, [14, 15], operator='>=', threshold=2),
    ]
    original, values = [0, 1, 0, 1, 2, 2, 9], [1, 3, 4, 7, 2, 5, 0]
    expected = [14, 15, 11, 11, 15, 15, 9]
    _check(original, values, expected, history, chunk_size)
    # Wrong rows remain detectable across chunk boundaries and after subsequent merges.
    for row in (0, 3, 6):
        wrong = expected.copy()
        wrong[row] = 15 if expected[row] != 15 else 14
        with pytest.raises(ValueError, match=f'at spike {row}$'):
            _check(original, values, wrong, history, chunk_size)


def test_reject_nonscalar_attribute():
    with pytest.raises(ValueError, match='scalar per source spike'):
        _check([0, 0], [[1, 2], [3, 4]], [11, 11], [_split(0, [10, 11])])
