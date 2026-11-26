"""Prepared-snapshot RP: offline memberships, boundaries, integrity and budgets."""

import json
from concurrent.futures import CancelledError
from shutil import copytree, rmtree

import numpy as np
import pytest
from click.testing import CliRunner

from phy.apps import phycli
from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.live_summaries import LiveSummaryEngine
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.refractory import RefractoryAccumulator
from phy.apps.large_recording.refractory_engine import RefractoryEngine
from phy.apps.large_recording.refractory_store import build_refractory_summaries
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.summary_store import (
    SummaryStore,
    attach_summary_store,
    build_summary_store,
    estimate_summary_store,
)
from phy.apps.large_recording.tests.test_refractory import brute
from phy.apps.large_recording.tests.test_refractory_engine import assert_oracle, make_package
from phy.apps.large_recording.tests.test_source import make_sorting


def no_source(*args, **kwargs):
    raise AssertionError('Prepared RP must not open/stat/hash the source.')


def equal_results(left, right):
    assert left.keys() == right.keys()
    for u in left:
        assert left[u].keys() == right[u].keys()
        for k in left[u]:
            if isinstance(left[u][k], np.ndarray):
                np.testing.assert_array_equal(left[u][k], right[u][k])
            else:
                assert left[u][k] == right[u][k]


@pytest.mark.parametrize('counters,timestamps', [(True, False), (False, True), (True, True)])
def test_originals_offline_equal_source_and_pairwise_oracle(tmp_path, counters, timestamps):
    source, bounded = make_package(tmp_path)
    expressions = {u: {'kind': 'source', 'unit': u} for u in (0, 1, 2, 99)}
    with LiveSummaryEngine(bounded, batch_size=2) as reader:
        expected = RefractoryEngine(reader).resolve(expressions)
    if counters:
        build_refractory_summaries(bounded, batch_size=2)
    if timestamps:
        build_summary_store(bounded, tmp_path / 'store', timestamps=True, batch_size=2)
    source.rename(tmp_path / 'offline')
    with LiveSummaryEngine(bounded, batch_size=1) as reader:
        reader._open = reader._unchanged = no_source
        engine = RefractoryEngine(reader)
        equal_results(engine.resolve(expressions), expected)
        equal_results(engine.resolve(expressions), expected)
        assert 'prepared' in engine.last_path


def test_timestamp_histories_oracle_undo_restore_reopen_and_portability(tmp_path):
    source, bounded = make_package(tmp_path)
    store = build_summary_store(bounded, tmp_path / 'store', timestamps=True, batch_size=2)
    original = {p.name: hash_file(p) for p in source.iterdir()}
    copytree(store, tmp_path / 'copied')
    rmtree(store)
    attach_summary_store(bounded, tmp_path / 'copied')
    with BoundedSession(bounded) as session, LiveSummaryEngine(bounded, batch_size=1) as reader:
        reader._open = no_source
        engine = RefractoryEngine(reader)
        assert_oracle(engine, session, source)
        session.record_merge([0, 1], 100)
        assert_oracle(engine, session, source)
        before = session.snapshot()
        session.record_threshold_split(
            100,
            [101, 102],
            {
                'attribute': 'amplitudes.npy',
                'operator': '>=',
                'threshold': -1 + 1e-12,
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
        session.save()
    with BoundedSession(bounded) as session, LiveSummaryEngine(bounded) as reader:
        reader._open = no_source
        assert_oracle(RefractoryEngine(reader), session, source)
    assert {p.name: hash_file(p) for p in source.iterdir()} == original


@pytest.mark.parametrize('batch_size', [1, 2, 5])
def test_scientific_and_custom_native_boundaries_precision_gaps(tmp_path, batch_size):
    params = make_sorting(tmp_path / 'source')
    edge = 1800 * 30000
    duration = 3 * edge + 1
    ticks = np.array(
        [
            edge - 20,
            edge - 9,
            edge - 1,
            edge,
            edge + 1,
            edge + 8,
            edge + 10,
            edge + 30,
            edge + 40,
            2 * edge,
            2 * edge + 10,
            3 * edge,
        ]
    )
    amplitudes = np.array([-1.0, 1 + 1e-12, 3.0, 2.0, -3.0, 4.0, 4.0, 2.0, 1.0, 3.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 99, 99], dtype=np.int32)
    for name, values in [
        ('spike_times', ticks),
        ('amplitudes', amplitudes),
        ('spike_clusters', labels),
    ]:
        np.save(params.parent / f'{name}.npy', values)
    params.write_text(
        params.read_text()
        .replace('sample_rate = 1.0', 'sample_rate = 30000.0')
        .replace('num_samples = 20', f'num_samples = {duration}')
    )
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        max_spikes=8,
        valid_intervals=[[0, 2 * edge + 20], [2 * edge + 100, duration]],
        bin_seconds=2700,
    )
    build_refractory_summaries(bounded, batch_size=batch_size)
    build_summary_store(bounded, tmp_path / 'store', timestamps=True, batch_size=batch_size)
    with LiveSummaryEngine(bounded, batch_size=batch_size) as source:
        source._open = no_source
        engine = RefractoryEngine(source)
        expressions = {
            0: {'kind': 'source', 'unit': 0},
            100: {'kind': 'union', 'parts': [{'kind': 'source', 'unit': u} for u in (0, 1)]},
        }
        results = engine.resolve(expressions)
        for unit, roots in [(0, [0]), (100, [0, 1])]:
            mask = np.isin(labels, roots)
            t, a = ticks[mask], amplitudes[mask]
            acc = RefractoryAccumulator(30000, duration, source.manifest['valid_intervals'])
            acc.feed(t, a)
            expected = acc.finish()
            equal_results({unit: results[unit]}, {unit: expected})
            kept = t[brute(t, a)]
            np.testing.assert_array_equal(
                results[unit]['spikes'], np.bincount(kept // edge, minlength=4)
            )
            v = np.zeros(4, dtype=np.int64)
            for i, x in enumerate(kept):
                v[x // edge] += sum(9 < int(y) - int(x) <= 30 for y in kept[i + 1 :])
            np.testing.assert_array_equal(results[unit]['violations'], v)


@pytest.mark.parametrize('artifact', ['counters', 'timestamps'])
def test_corruption_cancellation_and_cached_replacement_never_publish(tmp_path, artifact):
    source, bounded = make_package(tmp_path)
    if artifact == 'counters':
        path = build_refractory_summaries(bounded, batch_size=2)
        payload = path / '0.npz'
    else:
        path = build_summary_store(bounded, tmp_path / 'store', timestamps=True, batch_size=2)
        payload = path / 'data.bin'
    expression = {0: {'kind': 'source', 'unit': 0}}
    with LiveSummaryEngine(bounded, batch_size=1) as reader:
        engine = RefractoryEngine(reader)
        assert engine.resolve(expression)[0]['status'] == 'ready'
        with pytest.raises(CancelledError):
            engine.resolve(expression, cancel=lambda: True)
        raw = payload.read_bytes()
        payload.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        # The corrupted artifact cannot reuse its previously successful cache.
        reader._open = no_source
        with pytest.raises(AssertionError, match='must not'):
            engine.resolve(expression)
        if artifact == 'timestamps':
            assert engine.artifact_stamp is None
    assert hash_file(source / 'amplitudes.npy')


def test_timestamp_cancel_mid_read_has_no_partial_cache(tmp_path):
    _, bounded = make_package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store', timestamps=True, batch_size=1)
    cancelled = []
    with LiveSummaryEngine(bounded, batch_size=1) as reader:
        engine = RefractoryEngine(reader)
        with pytest.raises(CancelledError):
            engine.resolve(
                {0: {'kind': 'source', 'unit': 0}},
                lambda: bool(cancelled),
                lambda *args: cancelled.append(True),
            )
        assert not engine.results and not engine.prepared_results


def test_old_store_template_fallback_and_counter_budget(tmp_path):
    _, bounded = make_package(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        build_refractory_summaries(bounded, max_bytes=1)
    assert not (bounded / '.phy' / 'refractory-summaries').exists()
    build_summary_store(bounded, tmp_path / 'old')
    with LiveSummaryEngine(bounded) as reader:
        engine = RefractoryEngine(reader)
        expression = {
            'kind': 'split',
            'parent': {'kind': 'source', 'unit': 0},
            'predicate': {
                'attribute': 'spike_templates.npy',
                'operator': '>=',
                'threshold': 1,
                'nonfinite': 'keep',
            },
            'selected': True,
        }
        assert engine.resolve({1: expression})[1]['status'] == 'ready'
        assert engine.last_path == 'source scan'


def test_estimates_cli_and_compatible_scalar_consumers(tmp_path):
    _, bounded = make_package(tmp_path)
    runner = CliRunner()
    result = runner.invoke(phycli, ['build-rp-summaries', str(bounded), '--batch-size', '2'])
    assert result.exit_code == 0, result.output
    old = estimate_summary_store(bounded)
    new = estimate_summary_store(bounded, timestamps=True)
    assert new.payload_bytes > old.payload_bytes
    result = runner.invoke(
        phycli,
        [
            'build-summary-store',
            str(bounded),
            str(tmp_path / 'store'),
            '--timestamps',
            '--batch-size',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    with LiveSummaryEngine(bounded) as reader, SummaryStore(reader) as store:
        assert store.timestamps
        assert store.dtype['amplitude'] == np.dtype('float64')
        assert (
            sum(p.stat().st_size for p in store.path.iterdir() if p.is_file())
            <= new.estimated_bytes
        )
        expression = {0: {'kind': 'source', 'unit': 0}}
        # Existing rate/amplitude readers accept the extended row layout.
        reader._open = no_source
        assert reader.resolve(expression)[0]['counts'].sum() == 4


def test_prepare_optins_and_failure_preserve_bounded_package(tmp_path, monkeypatch):
    source, bounded = make_package(tmp_path)
    rmtree(bounded)
    prepared = prepare_large(
        source / 'params.py',
        bounded,
        rp_summaries=True,
        summary_store=tmp_path / 'rows',
        store_timestamps=True,
    )
    assert (prepared / '.phy' / 'refractory-summaries.json').exists()
    with LiveSummaryEngine(prepared) as reader, SummaryStore(reader) as store:
        assert store.timestamps
    with pytest.raises(ValueError, match='requires'):
        prepare_large(source / 'params.py', tmp_path / 'invalid', store_timestamps=True)
    assert not (tmp_path / 'invalid').exists()
    with pytest.raises(ValueError, match='disk budget.*--rp-summary-max-bytes'):
        prepare_large(
            source / 'params.py', tmp_path / 'budget', rp_summaries=True, rp_summary_max_bytes=1
        )
    assert not (tmp_path / 'budget').exists()

    def failed_build(*args, **kwargs):
        raise OSError('Simulated late write failure.')

    monkeypatch.setattr(
        'phy.apps.large_recording.refractory_store.build_refractory_summaries', failed_build
    )
    with pytest.raises(ValueError, match='Bounded package is ready.*build-rp-summaries'):
        prepare_large(source / 'params.py', tmp_path / 'late-failure', rp_summaries=True)
    with BoundedSession(tmp_path / 'late-failure') as session:
        assert session.unit_ids


def test_nonfinite_original_and_counters_cancel_are_isolated(tmp_path):
    source, bounded = make_package(tmp_path)
    rmtree(bounded)
    amp = np.load(source / 'amplitudes.npy')
    amp[0] = np.nan
    np.save(source / 'amplitudes.npy', amp)
    prepare_large(source / 'params.py', bounded)
    cancelled = []
    with pytest.raises(CancelledError):
        build_refractory_summaries(
            bounded,
            batch_size=1,
            cancel=lambda: bool(cancelled),
            progress=lambda *args: cancelled.append(True),
        )
    assert not (bounded / '.phy' / 'refractory-summaries.json').exists()
    assert not list((bounded / '.phy').glob('.rp-building-*'))
    build_refractory_summaries(bounded, batch_size=1)
    source.rename(tmp_path / 'offline')
    with LiveSummaryEngine(bounded) as reader:
        reader._open = no_source
        result = RefractoryEngine(reader).resolve(
            {u: {'kind': 'source', 'unit': u} for u in (0, 1)}
        )
        assert result[0]['status'] == 'unavailable'
        assert result[1]['status'] == 'ready'


def test_timestamp_offset_width_and_invalid_reconstruction(tmp_path):
    from phy.apps.large_recording.refractory_store import _unit_batches
    from phy.apps.large_recording.summary_store import StoreIntegrityError, _layout

    _, bounded = make_package(tmp_path)
    with LiveSummaryEngine(bounded) as reader:
        reader.edges = np.array([0, 2**32 + 1], dtype=np.int64)
        dtype, _, _ = _layout(reader, timestamps=True)
        assert dtype['tick_offset'] == np.dtype('uint64')
        reader.bins = 1

        class FakeStore:
            def batches(self, *args):
                data = np.zeros(1, dtype=dtype)
                data['tick_offset'] = 2**32 + 1
                yield data

        with pytest.raises(StoreIntegrityError, match='offset'):
            list(_unit_batches(FakeStore(), reader, 0, 1, None))


def test_timestamp_reference_changes_and_template_split_fallback(tmp_path):
    _, bounded = make_package(tmp_path)
    build_summary_store(bounded, tmp_path / 'first', timestamps=True)
    build_summary_store(bounded, tmp_path / 'second', timestamps=True, attach=False)
    with LiveSummaryEngine(bounded) as reader:
        engine = RefractoryEngine(reader)
        e = {0: {'kind': 'source', 'unit': 0}}
        engine.resolve(e)
        old_stamp = engine.artifact_stamp
        attach_summary_store(bounded, tmp_path / 'second')
        engine.resolve(e)
        assert old_stamp != engine.artifact_stamp
        expression = {
            'kind': 'split',
            'parent': e[0],
            'selected': True,
            'predicate': {
                'attribute': 'spike_templates.npy',
                'operator': '>=',
                'threshold': 1,
                'nonfinite': 'keep',
            },
        }
        assert engine.resolve({10: expression})[10]['status'] == 'ready'
        assert engine.last_path == 'source scan'


@pytest.mark.parametrize('counters', [False, True])
def test_result_grid_budget_is_explicit_and_never_reads_source(tmp_path, counters):
    _, bounded = make_package(tmp_path)
    if counters:
        build_refractory_summaries(bounded)
    else:
        build_summary_store(bounded, tmp_path / 'store', timestamps=True)
    with LiveSummaryEngine(bounded) as reader:
        reader._open = no_source
        value = RefractoryEngine(reader, result_bytes=1).resolve(
            {0: {'kind': 'source', 'unit': 0}}
        )[0]
        assert value['status'] == 'unavailable'
        assert 'budget' in value['message']


def test_offline_unsupported_child_preserves_available_original(tmp_path):
    source, bounded = make_package(tmp_path)
    build_refractory_summaries(bounded)
    source.rename(tmp_path / 'offline')
    with LiveSummaryEngine(bounded) as reader:
        result = RefractoryEngine(reader).resolve(
            {
                0: {'kind': 'source', 'unit': 0},
                100: {
                    'kind': 'split',
                    'parent': {'kind': 'source', 'unit': 1},
                    'selected': True,
                    'predicate': {
                        'attribute': 'spike_templates.npy',
                        'operator': '>=',
                        'threshold': 1,
                        'nonfinite': 'keep',
                    },
                },
            }
        )
        assert result[0]['status'] == 'ready'
        assert result[100]['status'] == 'unavailable'


def test_counter_publication_failure_is_retryable_without_partial_results(tmp_path, monkeypatch):
    from phy.apps.large_recording import refractory_store

    _, bounded = make_package(tmp_path)
    publish = refractory_store.publish_directory

    def failed(*args):
        raise OSError('injected publication failure')

    monkeypatch.setattr(refractory_store, 'publish_directory', failed)
    with pytest.raises(OSError, match='injected'):
        build_refractory_summaries(bounded)
    assert not (bounded / '.phy' / 'refractory-summaries').exists()
    with LiveSummaryEngine(bounded) as reader:
        engine = RefractoryEngine(reader)
        assert engine.resolve({0: {'kind': 'source', 'unit': 0}})[0]['status'] == 'ready'
        assert engine.last_path == 'source scan'
    monkeypatch.setattr(refractory_store, 'publish_directory', publish)
    build_refractory_summaries(bounded)
    with LiveSummaryEngine(bounded) as reader:
        reader._open = no_source
        assert (
            RefractoryEngine(reader).resolve({0: {'kind': 'source', 'unit': 0}})[0]['status']
            == 'ready'
        )


def test_reattachment_during_store_open_never_publishes_old_result(tmp_path, monkeypatch):
    from phy.apps.large_recording import refractory_store

    _, bounded = make_package(tmp_path)
    build_summary_store(bounded, tmp_path / 'first', timestamps=True)
    build_summary_store(bounded, tmp_path / 'second', timestamps=True, attach=False)
    original = refractory_store.SummaryStore

    def racing_open(reader):
        store = original(reader)
        attach_summary_store(bounded, tmp_path / 'second')
        return store

    monkeypatch.setattr(refractory_store, 'SummaryStore', racing_open)
    with LiveSummaryEngine(bounded) as reader:
        reader._open = no_source
        engine = RefractoryEngine(reader)
        with pytest.raises(AssertionError, match='must not'):
            engine.resolve({0: {'kind': 'source', 'unit': 0}})
        assert engine.artifact_stamp is None
        assert not engine.prepared_results


def test_timestamp_empty_child_and_uint64_offsets_are_exact(tmp_path):
    from phy.apps.large_recording.refractory_store import _unit_batches

    _, bounded = make_package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store', timestamps=True)
    expression = {
        'kind': 'split',
        'parent': {'kind': 'source', 'unit': 0},
        'selected': True,
        'predicate': {
            'attribute': 'amplitudes.npy',
            'operator': '>=',
            'threshold': 1000,
            'nonfinite': 'keep',
        },
    }
    with LiveSummaryEngine(bounded) as reader:
        reader._open = no_source
        result = RefractoryEngine(reader).resolve({100: expression})[100]
        assert result['status'] == 'ready'
        assert result['spikes'].sum() == result['violations'].sum() == 0
        assert np.isnan(result['contamination']).all()
        reader.edges = np.array([0, 2**32 + 1], dtype=np.int64)
        reader.bins = 1

        class FakeStore:
            def batches(self, *args):
                data = np.zeros(
                    1, dtype=[('bin', 'u1'), ('amplitude', 'f8'), ('tick_offset', 'u8')]
                )
                data['tick_offset'] = 2**32
                yield data

        rows = list(_unit_batches(FakeStore(), reader, 0, 1, None))
        np.testing.assert_array_equal(rows[0]['spike_times.npy'], [2**32])
