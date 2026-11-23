"""Current-membership, cancellation, source-integrity and cache regressions."""

from concurrent.futures import CancelledError

import numpy as np
import pytest

from phy.apps.large_recording.curation import BoundedSession, project_assignments
from phy.apps.large_recording.live_summaries import (
    LiveSummaryEngine,
    acquire_source_engine,
    membership_expressions,
    release_source_engine,
)
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.refractory_engine import RefractoryEngine
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.tests.test_refractory import brute
from phy.apps.large_recording.tests.test_source import make_sorting


def make_package(tmp_path):
    params = make_sorting(tmp_path / 'source')
    params.write_text(params.read_text().replace('sample_rate = 1.0', 'sample_rate = 30000.0'))
    amp = np.arange(12, dtype=np.float64) - 6
    np.save(params.parent / 'amplitudes.npy', amp)
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        max_spikes=8,
        window_seconds=0.0001,
        stratum_seconds=0.0002,
        bin_seconds=0.0002,
    )
    return params.parent, bounded


def assert_oracle(engine, session, source):
    expressions = session.membership_snapshot(session._runtime.active)
    assert expressions == membership_expressions(session.unit_ids, session.events)
    labels = np.load(source / 'spike_clusters.npy')
    amp = np.load(source / 'amplitudes.npy')
    labels = project_assignments(labels, session.events, {'amplitudes.npy': amp})
    ticks = np.load(source / 'spike_times.npy')
    results = engine.resolve(expressions)
    for unit, r in results.items():
        selected = labels == unit
        t, a = ticks[selected], amp[selected]
        kept = t[brute(t, a)]
        assert r['spikes'].sum() == len(kept)
        assert r['duplicates_removed'] == len(t) - len(kept)
        assert r['violations'].sum() == sum(
            9 < int(y) - int(x) <= 30 for i, x in enumerate(kept) for y in kept[i + 1 :]
        )


def test_live_membership_restore_undo_reopen_and_signed_predicates(tmp_path):
    source, bounded = make_package(tmp_path)
    original = {p.name: hash_file(p) for p in source.iterdir()}
    with BoundedSession(bounded) as session, LiveSummaryEngine(bounded, batch_size=3) as reader:
        engine = RefractoryEngine(reader)
        assert_oracle(engine, session, source)
        session.record_merge([0, 1], 100)
        assert_oracle(engine, session, source)
        assert engine.last_path == 'unit ticks'
        frozen = session.membership_snapshot([100])
        before = session.snapshot()
        session.record_threshold_split(
            100,
            [101, 102],
            {
                'attribute': 'amplitudes.npy',
                'operator': '>=',
                'threshold': 0,
                'nonfinite': 'keep',
            },
        )
        assert_oracle(engine, session, source)
        after = session.snapshot()
        session.restore(before)
        assert_oracle(engine, session, source)
        session.restore(after)
        assert_oracle(engine, session, source)
        session.record_merge([101, 102], 103)
        assert_oracle(engine, session, source)
        session.record_restore_parents(103, [104, 105], recursive=True)
        assert_oracle(engine, session, source)
        session.record_merge([104, 2], 106)
        assert_oracle(engine, session, source)
        with pytest.raises(TypeError):
            frozen[100]['kind'] = 'source'
        session.save()
    with BoundedSession(bounded) as reopened, LiveSummaryEngine(bounded) as reader:
        assert_oracle(RefractoryEngine(reader), reopened, source)
    assert {p.name: hash_file(p) for p in source.iterdir()} == original


def test_cancel_reuses_authenticated_readers_and_rejects_warm_mutation(tmp_path):
    source, bounded = make_package(tmp_path)
    expressions = {0: {'kind': 'source', 'unit': 0}}
    with LiveSummaryEngine(bounded, batch_size=3) as reader:
        engine = RefractoryEngine(reader)
        engine.resolve(expressions)
        handles = dict(reader.handles)
        with pytest.raises(CancelledError):
            engine.resolve({1: {'kind': 'source', 'unit': 1}}, lambda: True)
        assert reader.handles == handles
        assert not any(h.closed for h in handles.values())
        assert engine.resolve(expressions)[0]['spikes'].sum() > 0
        amp = np.load(source / 'amplitudes.npy', mmap_mode='r+')
        amp[0] += 1
        amp.flush()
        with pytest.raises(ValueError, match='changed'):
            engine.resolve(expressions)


def test_root_budget_does_not_retain_oversized_rows(tmp_path):
    _, bounded = make_package(tmp_path)
    with LiveSummaryEngine(bounded, batch_size=2) as reader:
        engine = RefractoryEngine(reader, root_bytes=1)
        engine.resolve({0: {'kind': 'source', 'unit': 0}})
        assert not engine.roots


def test_two_clients_share_authentication_and_only_final_close_releases(tmp_path):
    _, bounded = make_package(tmp_path)
    first_key, first = acquire_source_engine(bounded)
    second_key, second = acquire_source_engine(bounded)
    assert first is second
    try:
        engine = RefractoryEngine(first)
        engine.resolve({0: {'kind': 'source', 'unit': 0}})
        handles = dict(first.handles)
        release_source_engine(first_key)
        first_key = None
        assert not any(h.closed for h in handles.values())
        second._open(set(handles), None)
        assert second.handles == handles
        with pytest.raises(CancelledError):
            second._open(set(handles), lambda: True)
        assert second.handles == handles
    finally:
        if first_key is not None:
            release_source_engine(first_key)
        release_source_engine(second_key)
    assert all(h.closed for h in handles.values())


def test_cancel_mid_scan_retains_trust_but_never_partial_result(tmp_path):
    _, bounded = make_package(tmp_path)
    with LiveSummaryEngine(bounded, batch_size=2) as reader:
        engine = RefractoryEngine(reader)
        engine.resolve({0: {'kind': 'source', 'unit': 0}})
        handles = dict(reader.handles)
        cancelled = []
        with pytest.raises(CancelledError):
            engine.resolve(
                {1: {'kind': 'source', 'unit': 1}},
                lambda: bool(cancelled),
                lambda *args: cancelled.append(True),
            )
        assert reader.handles == handles
        assert engine.key({'kind': 'source', 'unit': 1}) not in engine.results
        assert engine.resolve({1: {'kind': 'source', 'unit': 1}})[1]['status'] == 'ready'
        assert reader.handles == handles


def test_cancelled_partial_authentication_keeps_metadata_mutation_guard(tmp_path):
    source, bounded = make_package(tmp_path)
    with LiveSummaryEngine(bounded) as reader:
        with pytest.raises(CancelledError):
            reader._open(
                {'spike_clusters.npy', 'spike_times.npy'},
                lambda: 'spike_clusters.npy' in reader.arrays,
            )
        assert 'spike_clusters.npy' in reader.arrays
        with (source / 'cluster_group.tsv').open('a') as stream:
            stream.write('100\tgood\n')
        with pytest.raises(ValueError, match='metadata changed'):
            reader._open({'spike_clusters.npy'}, None)


def test_one_nonfinite_unit_does_not_hide_other_rp_results(tmp_path):
    source, bounded = make_package(tmp_path)
    # Modify before preparation so the evidence identity authentically includes NaN.
    from shutil import rmtree

    rmtree(bounded)
    amp = np.load(source / 'amplitudes.npy')
    amp[0] = np.nan
    np.save(source / 'amplitudes.npy', amp)
    bounded = prepare_large(
        source / 'params.py',
        bounded,
        max_spikes=8,
        window_seconds=0.0001,
        stratum_seconds=0.0002,
        bin_seconds=0.0002,
    )
    with LiveSummaryEngine(bounded) as reader:
        results = RefractoryEngine(reader).resolve(
            {u: {'kind': 'source', 'unit': u} for u in (0, 1)}
        )
    assert results[0]['status'] == 'unavailable'
    assert results[1]['status'] == 'ready'
