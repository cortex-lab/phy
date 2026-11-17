"""Independently verify a split/merge/restore/remerge against original source rows.

The scenario records user-visible operands and resulting IDs, not production
membership expressions. Expected assignments use only original labels and raw
finite amplitude comparisons; no Phy journal/projector is imported.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def verify(source, outputs, scenario, batch_size=1_000_000):
    labels = np.load(source / 'spike_clusters.npy', mmap_mode='r').reshape(-1)
    amplitudes = np.load(source / 'amplitudes.npy', mmap_mode='r').reshape(-1)
    actual = [np.load(path / 'spike_clusters.npy', mmap_mode='r').reshape(-1) for path in outputs]
    assert len(labels) == len(amplitudes)
    assert all(len(values) == len(labels) for values in actual)
    assert np.isfinite(scenario['threshold']) and batch_size > 0
    counts = Counter()
    changed = 0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        original = labels[start:stop]
        values = amplitudes[start:stop]
        family = np.isin(original, scenario['initial_parents'])
        below = np.isfinite(values) & (values < scenario['threshold'])
        expected = original.copy()
        expected[family & ~below] = scenario['split_remainder']
        expected[family & below] = scenario['final_merged']
        expected[original == scenario['excluded_original']] = scenario['excluded_live']
        expected[original == scenario['new_candidate']] = scenario['final_merged']
        expected[np.isin(original, scenario['unrelated_inputs'])] = scenario['unrelated_result']
        for output in actual:
            np.testing.assert_array_equal(output[start:stop], expected)
        ids, frequencies = np.unique(expected, return_counts=True)
        counts.update(dict(zip(map(int, ids), map(int, frequencies))))
        changed += int(np.count_nonzero(expected != original))
    return {
        'rows_checked_per_output': len(labels),
        'outputs': len(outputs),
        'assignments_exact': True,
        'changed_rows': changed,
        'active_units': len(counts),
        'counts': dict(sorted(counts.items())),
        'oracle': 'Original source labels plus raw finite amplitude < threshold; no Phy imports',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1_000_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    result = verify(
        args.source, args.outputs, json.loads(args.scenario.read_text()), args.batch_size
    )
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'counts'}, indent=2))
