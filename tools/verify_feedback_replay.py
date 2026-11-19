"""Independently check a demonstrated merge followed by an amplitude split.

Reads only NumPy arrays, without Phy's journal/projection implementation. The
two children are the finite-below-threshold selection and the remainder. All
source units outside the supplied parent union must remain unchanged.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def verify(args):
    labels = np.load(args.source / 'spike_clusters.npy', mmap_mode='r').reshape(-1)
    amplitudes = np.load(args.source / 'amplitudes.npy', mmap_mode='r').reshape(-1)
    outputs = [
        np.load(path / 'spike_clusters.npy', mmap_mode='r').reshape(-1) for path in args.outputs
    ]
    assert len(amplitudes) == len(labels)
    assert all(len(output) == len(labels) for output in outputs)
    counts = Counter()
    for start in range(0, len(labels), args.batch_size):
        stop = min(start + args.batch_size, len(labels))
        original = labels[start:stop]
        values = amplitudes[start:stop]
        parent = np.isin(original, args.parents)
        selected = np.isfinite(values) & (values < args.threshold)
        expected = original.copy()
        expected[parent & selected] = args.children[0]
        expected[parent & ~selected] = args.children[1]
        for output in outputs:
            np.testing.assert_array_equal(output[start:stop], expected)
        ids, frequencies = np.unique(expected, return_counts=True)
        counts.update(dict(zip(map(int, ids), map(int, frequencies))))
    result = {
        'rows_checked_per_output': len(labels),
        'outputs': len(outputs),
        'assignments_exact': True,
        'active_units': len(counts),
        'children': {str(unit): counts[unit] for unit in args.children},
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--parents', type=int, nargs='+', required=True)
    parser.add_argument('--children', type=int, nargs=2, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=1_000_000)
    args = parser.parse_args()
    if args.batch_size < 1 or not np.isfinite(args.threshold):
        parser.error('Use a positive batch size and a finite threshold.')
    verify(args)
