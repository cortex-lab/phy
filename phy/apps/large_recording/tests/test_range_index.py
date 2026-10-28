"""Range acceleration is an exact optional optimization of full curation replay."""

import numpy as np
import pytest

from phy.apps.large_recording.live_summaries import LiveSummaryEngine, membership_expressions
from phy.apps.large_recording.range_build import build_amplitude_index
from phy.apps.large_recording.range_index import AmplitudeRangeIndex
from phy.apps.large_recording.summary_store import build_summary_store
from phy.apps.large_recording.tests.test_live_summaries import merge, oracle, package, split


@pytest.mark.parametrize('block_rows', [1, 2, 3, 128])
@pytest.mark.parametrize('columns,positions', [(False, False), (True, True)])
def test_range_queries_match_independent_full_replay(tmp_path, block_rows, columns, positions):
    source, bounded = package(tmp_path, columns, positions)
    build_summary_store(bounded, tmp_path / 'store', batch_size=2)
    build_amplitude_index(bounded, tmp_path / 'ranges', block_rows=block_rows)
    with LiveSummaryEngine(bounded) as engine:
        index = AmplitudeRangeIndex(engine)
        try:
            for operator in ['<', '<=', '>', '>=']:
                events = [split(0, [10, 11]), merge([10, 1], 12), split(12, [13, 14], threshold=6)]
                events[-1]['predicate']['operator'] = operator
                expressions = membership_expressions([0, 1, 2, 99], events)
                actual = index.resolve(expressions)
                for unit, summary in actual.items():
                    expected = oracle(source, engine, events, unit)
                    for name, values in engine._flat(summary).items():
                        np.testing.assert_allclose(
                            values, engine._flat(expected)[name], rtol=1e-9, atol=1e-6
                        )
        finally:
            index.close()


def assert_summary(engine, actual, expected):
    for name, values in engine._flat(actual).items():
        target = engine._flat(expected)[name]
        if np.issubdtype(values.dtype, np.integer):
            np.testing.assert_array_equal(values, target)
        else:
            np.testing.assert_allclose(values, target, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'uint64'])
def test_dtype_boundaries_and_nonfinite_bins_match_replay(tmp_path, dtype):
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.tests.test_source import make_sorting

    params = make_sorting(tmp_path / 'source')
    values = np.array([0, 1, 1, 2, 2, 3, 1, 4, 5, 6, 8, 9], dtype=dtype)
    if dtype.startswith('float'):
        values[[1, 4]] = [np.nan, np.inf]  # all-nonfinite unit/bin with finite depth
    else:
        values[-3:] = [2**53 + 1, 2**63 - 1, 2**63 - 1 if dtype == 'int64' else 2**64 - 1]
    np.save(params.parent / 'amplitudes.npy', values)
    bounded = prepare_large(params, tmp_path / 'bounded', bin_seconds=5, batch_size=2)
    build_summary_store(bounded, tmp_path / 'store')
    build_amplitude_index(bounded, tmp_path / 'ranges', block_rows=2)
    with LiveSummaryEngine(bounded) as engine:
        index = AmplitudeRangeIndex(engine)
        try:
            for threshold in [1.000000001, float(2**63), 2**53 + 1, -999, 1e30]:
                for operator in ['<', '<=', '>', '>=']:
                    events = [merge([0, 1, 99], 100), split(100, [101, 102], threshold=threshold)]
                    events[-1]['predicate']['operator'] = operator
                    expressions = membership_expressions(engine.manifest['unit_ids'], events)
                    expressions = {u: expressions[u] for u in [101, 102]}
                    actual = index.resolve(expressions)
                    for unit in expressions:
                        assert_summary(
                            engine, actual[unit], oracle(params.parent, engine, events, unit)
                        )
        finally:
            index.close()


def test_external_sort_is_stable_and_bounded_with_duplicate_keys(tmp_path):
    from phy.apps.large_recording.range_build import _SortedRuns

    rng = np.random.default_rng(8)
    values = np.empty(10000, dtype=[('amplitude', 'f4'), ('depth', 'i8')])
    values['amplitude'] = rng.integers(-100, 100, len(values))
    values['depth'] = np.arange(len(values))
    sorter = _SortedRuns(tmp_path, values.dtype, 2048, None)
    for first in range(0, len(values), 37):
        sorter.add(values[first : first + 37])
    records, path = sorter.finish()
    assert records is None
    actual = np.fromfile(path, dtype=values.dtype)
    np.testing.assert_array_equal(actual, values[np.argsort(values['amplitude'], kind='stable')])
    path.unlink()
    assert not list(tmp_path.iterdir())


def test_unstable_arithmetic_uses_verified_rows_and_reject_still_raises(tmp_path, monkeypatch):
    from phy.apps.large_recording import range_index
    from phy.apps.large_recording.range_intervals import RangeUnavailableError

    source, bounded = package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store')
    build_amplitude_index(bounded, tmp_path / 'ranges')

    def unstable(*args):
        raise RangeUnavailableError('unstable arithmetic')

    monkeypatch.setattr(range_index, 'validate_sums', unstable)
    with LiveSummaryEngine(bounded) as engine:
        engine._publish = lambda *args: None
        events = [split(0, [100, 101])]
        expressions = membership_expressions(engine.manifest['unit_ids'], events)
        actual = engine.resolve({u: expressions[u] for u in [100, 101]})
        assert engine.range_index is not None
        assert not engine.arrays  # fallback reads the verified rows, not full-source arrays
        for unit, summary in actual.items():
            assert_summary(engine, summary, oracle(source, engine, events, unit))
        events = [split(1, [100, 101], nonfinite='reject')]
        expressions = membership_expressions(engine.manifest['unit_ids'], events)
        with pytest.raises(ValueError, match='Nonfinite'):
            engine.resolve({100: expressions[100]})


def test_corrupt_accelerator_falls_back_and_reattachment_invalidates_readers(tmp_path):
    from phy.apps.large_recording.range_index import attach_amplitude_index
    from phy.apps.large_recording.summary_store import attach_summary_store

    source, bounded = package(tmp_path)
    store = build_summary_store(bounded, tmp_path / 'store')
    ranges = build_amplitude_index(bounded, tmp_path / 'ranges')
    events = [split(0, [100, 101])]
    with LiveSummaryEngine(bounded) as engine:
        engine._publish = lambda *args: None
        e = membership_expressions(engine.manifest['unit_ids'], events)
        e = {u: e[u] for u in [100, 101]}
        engine.resolve(e)
        previous = engine.range_index
        with (ranges / 'nodes.bin').open('r+b') as stream:
            stream.write(b'x')
        actual = engine.resolve(e)
        assert engine.range_index is None
        assert previous.files == {}
        for unit in e:
            assert_summary(engine, actual[unit], oracle(source, engine, events, unit))
        replacement = build_amplitude_index(bounded, tmp_path / 'replacement')
        actual = engine.resolve(e)
        assert engine.range_index.path == replacement
        attach_summary_store(bounded, store)
        engine.resolve(e)
        assert engine.range_index is None
        attach_amplitude_index(bounded, replacement)
        engine.resolve(e)
        assert engine.range_index is not None


def test_cancel_budget_and_automatic_attachment_conflict_leave_recoverable_artifacts(tmp_path):
    from concurrent.futures import CancelledError

    from phy.apps.large_recording.range_index import attach_amplitude_index
    from phy.apps.large_recording.summary_store import (
        SummaryStorePublishedError,
        summary_store_reference,
    )

    _, bounded = package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store')
    reference = summary_store_reference(bounded)
    with pytest.raises(ValueError, match='max_bytes'):
        build_amplitude_index(bounded, tmp_path / 'too-big', max_bytes=1)
    with pytest.raises(CancelledError):
        build_amplitude_index(bounded, tmp_path / 'cancelled', cancel=lambda: True)
    assert not list(tmp_path.glob('.*.building-*'))
    assert summary_store_reference(bounded) == reference
    first = build_amplitude_index(bounded, tmp_path / 'first', attach=False)

    def choose_first():
        attach_amplitude_index(bounded, first)

    with pytest.raises(SummaryStorePublishedError, match='changed'):
        build_amplitude_index(bounded, tmp_path / 'second', before_publish=choose_first)
    assert (tmp_path / 'second' / 'range.json').is_file()
    with LiveSummaryEngine(bounded) as engine:
        index = AmplitudeRangeIndex(engine)
        assert index.path == first
        index.close()


def test_ill_conditioned_depth_falls_back_without_changing_source_order_sum(tmp_path):
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.range_intervals import RangeUnavailableError
    from phy.apps.large_recording.tests.test_source import make_sorting

    params = make_sorting(tmp_path / 'source')
    depth = np.zeros(12)
    depth[[0, 3, 6, 9]] = [1e20, 1, -1e20, 0]
    np.save(params.parent / 'spike_depths.npy', depth)
    bounded = prepare_large(params, tmp_path / 'bounded', bin_seconds=20)
    build_summary_store(bounded, tmp_path / 'store')
    build_amplitude_index(bounded, tmp_path / 'ranges', block_rows=2)
    events = [split(0, [100, 101], threshold=-1)]
    with LiveSummaryEngine(bounded) as engine:
        expressions = membership_expressions(engine.manifest['unit_ids'], events)
        expressions = {101: expressions[101]}
        index = AmplitudeRangeIndex(engine)
        try:
            with pytest.raises(RangeUnavailableError, match='arithmetic'):
                index.resolve(expressions)
        finally:
            index.close()
        actual = engine.resolve(expressions)
        assert not engine.arrays
        assert_summary(engine, actual[101], oracle(params.parent, engine, events, 101))


@pytest.mark.parametrize('stage', ['sort', 'hash', 'publish'])
def test_failed_build_cleans_staging_without_changing_attachment(tmp_path, monkeypatch, stage):
    from phy.apps.large_recording import range_build
    from phy.apps.large_recording.summary_store import summary_store_reference

    _, bounded = package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store')
    previous = summary_store_reference(bounded)

    def fail(*args, **kwargs):
        raise OSError('injected disk failure')

    if stage == 'sort':
        monkeypatch.setattr(range_build._SortedRuns, 'finish', fail)
    elif stage == 'hash':
        monkeypatch.setattr(range_build, 'authenticate_file', fail)
    else:
        monkeypatch.setattr(range_build, 'publish_directory', fail)
    with pytest.raises(OSError, match='injected'):
        build_amplitude_index(bounded, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('.*.building-*'))
    assert summary_store_reference(bounded) == previous
