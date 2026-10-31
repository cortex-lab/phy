"""Selective authentication and exact interval/threshold semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from phy.apps.large_recording.curation import evaluate_predicate, project_assignments
from phy.apps.large_recording.live_summaries import membership_expressions
from phy.apps.large_recording.range_files import (
    PAGE_BYTES,
    PageCache,
    VerifiedFile,
    authenticate_file,
)
from phy.apps.large_recording.range_intervals import IntervalProgram
from phy.apps.large_recording.range_math import typed_rank
from phy.apps.large_recording.summary_store import StoreIntegrityError
from phy.apps.large_recording.tests.test_live_summaries import merge, split


@pytest.mark.parametrize('length', [0, 1, 64, 4096, 4097, 100000])
def test_verified_random_reads_and_bounded_cache(tmp_path, length):
    raw = np.random.default_rng(1).integers(0, 256, length, dtype=np.uint8).tobytes()
    path = tmp_path / 'values.bin'
    path.write_bytes(raw)
    descriptor = authenticate_file(path)
    cache = PageCache(2 * PAGE_BYTES)
    reader = VerifiedFile(path, descriptor, cache)
    try:
        for start in range(0, length, 1137):
            assert reader.read(start, min(777, length - start)) == raw[start : start + 777]
        reader.unchanged()
        assert cache.size <= cache.max_bytes
    finally:
        reader.close()


def test_corruption_is_selective_and_detected_before_use(tmp_path):
    path = tmp_path / 'data.bin'
    path.write_bytes(bytes(range(256)) * 1000)
    descriptor = authenticate_file(path)
    with path.open('r+b') as stream:
        stream.seek(40 * PAGE_BYTES + 2)
        stream.write(b'x')
    reader = VerifiedFile(path, descriptor, PageCache(8192))
    try:
        assert reader.read(0, 8) == bytes(range(8))
        assert reader.bytes_read == PAGE_BYTES  # No whole-file integrity scan.
        with pytest.raises(StoreIntegrityError, match='checksum'):
            reader.read(40 * PAGE_BYTES, 16)
    finally:
        reader.close()


def test_changed_pinned_file_and_hash_tree_corruption(tmp_path):
    path = tmp_path / 'data.bin'
    path.write_bytes(b'a' * 10000)
    descriptor = authenticate_file(path)
    reader = VerifiedFile(path, descriptor, PageCache(8192))
    try:
        with path.open('r+b') as stream:
            stream.write(b'b')
        with pytest.raises(StoreIntegrityError, match='changed'):
            reader.unchanged()
    finally:
        reader.close()
    authenticate_file(path)
    hashes = path.with_suffix('.bin.hashes')
    with hashes.open('r+b') as stream:
        stream.seek(32)
        stream.write(b'x' * 32)
    with pytest.raises(StoreIntegrityError, match='root'):
        VerifiedFile(path, descriptor, PageCache(8192))


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'uint64'])
@pytest.mark.parametrize('operator', ['<', '<='])
def test_rank_boundaries_match_canonical_array_comparisons(dtype, operator):
    values = np.array([0, 1, 1, 2, 7], dtype=dtype)
    if dtype in ('int64', 'uint64'):
        values = np.r_[values, np.array([2**53, 2**53 + 1, 2**63 - 1], dtype=dtype)]
    for threshold in [
        -1,
        0,
        1,
        1.000000001,
        float(np.nextafter(np.float32(1), np.float32(2))),
        2**53,
        float(2**53),
        1e30,
    ]:
        predicate = {
            'attribute': 'amplitudes.npy',
            'operator': operator,
            'threshold': threshold,
            'nonfinite': 'keep',
        }
        assert typed_rank(values, predicate) == int(evaluate_predicate(values, predicate).sum())


def test_interval_graph_matches_replay_for_nested_merges_and_all_comparators():
    original = np.repeat([0, 1, 2], 12)
    amplitude = np.tile(np.array([-np.inf, -2, -0.0, 0.0, 1, 1, 2, 3, 4, 8, np.inf, np.nan]), 3)
    for operator in ['<', '<=', '>', '>=']:
        events = [
            split(0, [10, 11], threshold=2),
            merge([10, 1], 12),
            split(12, [13, 14], threshold=1),
        ]
        events[-1]['predicate']['operator'] = operator
        expressions = membership_expressions([0, 1, 2], events)
        program = IntervalProgram(expressions)
        expected = project_assignments(original, events, {'amplitudes.npy': amplitude})
        for source in [0, 1, 2]:
            rows = np.flatnonzero(original == source)
            finite = rows[np.isfinite(amplitude[rows])]
            finite = finite[np.argsort(amplitude[finite], kind='stable')]
            nonfinite = rows[~np.isfinite(amplitude[rows])]
            cuts = {key: typed_rank(amplitude[finite], pred) for key, pred in program.cuts.items()}
            actual = program.evaluate(source, len(finite), len(nonfinite), cuts)
            for unit, (parts, include_nonfinite) in actual.items():
                selected = [int(i) for lo, hi in parts for i in finite[lo:hi]]
                if include_nonfinite:
                    selected.extend(nonfinite.tolist())
                assert (
                    sorted(selected)
                    == np.flatnonzero((original == source) & (expected == unit)).tolist()
                )


def test_reject_obligation_survives_empty_descendant_and_unrelated_units():
    events = [
        split(0, [10, 11], threshold=2, nonfinite='reject'),
        split(11, [12, 13], threshold=-999),
    ]
    expressions = membership_expressions([0, 1], events)
    program = IntervalProgram({13: expressions[13]})
    cuts = dict.fromkeys(program.cuts, 0)
    with pytest.raises(ValueError, match='Nonfinite'):
        program.evaluate(0, 5, 1, cuts)
    assert program.evaluate(1, 5, 1, cuts)[13] == ([], False)


def test_root_remains_anchor_after_node_cache_eviction(tmp_path):
    path = tmp_path / 'data.bin'
    path.write_bytes(b'a' * PAGE_BYTES + b'b' * PAGE_BYTES)
    reader = VerifiedFile(path, authenticate_file(path), PageCache(4096))
    try:
        reader.nodes.clear()
        assert reader.read(PAGE_BYTES, 4) == b'bbbb'
    finally:
        reader.close()
