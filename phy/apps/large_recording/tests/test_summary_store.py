"""Unit-contiguous snapshot construction and selective integrity validation."""

import json
from concurrent.futures import CancelledError

import numpy as np
import pytest

from phy.apps.large_recording.live_summaries import LiveSummaryEngine
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.summary_store import SummaryStore, _bin_dtype, build_summary_store
from phy.apps.large_recording.tests.test_live_summaries import package


@pytest.mark.parametrize('columns,positions', [(False, False), (True, True)])
def test_store_exact_values_and_package_unchanged(tmp_path, columns, positions):
    source, bounded = package(tmp_path, columns, positions)
    before = {p.name: hash_file(p) for p in bounded.iterdir() if p.is_file()}
    output = build_summary_store(bounded, tmp_path / 'store', batch_size=2, buffer_bytes=12)
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        assert store.dtype['bin'] == np.dtype('uint8')
        assignments = np.load(source / 'spike_clusters.npy').reshape(-1)
        times = np.load(source / 'spike_times.npy').reshape(-1)
        for unit in engine.manifest['unit_ids']:
            data = np.concatenate(list(store.batches(unit, 2)))
            selected = assignments == unit
            np.testing.assert_array_equal(
                data['bin'], np.searchsorted(engine.edges, times[selected], side='right') - 1
            )
            amps = np.load(source / 'amplitudes.npy').reshape(-1)
            np.testing.assert_array_equal(data['amplitude'], amps[selected])
            np.testing.assert_array_equal(data['depth'], times[selected] * 2)
    assert output.is_dir()
    assert {p.name: hash_file(p) for p in bounded.iterdir() if p.is_file()} == before
    assert not list(tmp_path.glob('.*.building-*'))


def test_store_budgets_cancel_and_failure_do_not_publish(tmp_path):
    _, bounded = package(tmp_path)
    output = tmp_path / 'store'
    with pytest.raises(ValueError, match='increase max_bytes'):
        build_summary_store(bounded, output, max_bytes=1)
    with pytest.raises(ValueError, match='outside'):
        build_summary_store(bounded, bounded / 'index')

    def progress(phase, done, total):
        if phase == 'building summary store':
            raise CancelledError()

    with pytest.raises(CancelledError):
        build_summary_store(bounded, output, batch_size=2, progress=progress)
    assert not output.exists()
    assert not (bounded / '.phy' / 'live-summary-store.json').exists()
    assert not list(tmp_path.glob('.*.building-*'))
    build_summary_store(bounded, output)
    with pytest.raises(ValueError, match='already exists'):
        build_summary_store(bounded, output)


def test_store_selective_hash_and_corruption(tmp_path):
    _, bounded = package(tmp_path)
    output = build_summary_store(bounded, tmp_path / 'store')
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        item = store.index['units']['1']
        offset = item['start'] * store.dtype.itemsize
    with (output / 'data.bin').open('r+b') as stream:
        stream.seek(offset)
        byte = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([byte[0] ^ 1]))
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        list(store.batches(0, 2))  # An unrelated unit remains independently verifiable.
        with pytest.raises(ValueError, match='checksum'):
            list(store.batches(1, 2))
    index = output / 'index.json'
    catalog = json.loads(index.read_text())
    catalog['units']['0']['count'] += 1
    index.write_text(json.dumps(catalog))
    with LiveSummaryEngine(bounded) as engine, pytest.raises(ValueError, match='binding'):
        SummaryStore(engine)


def test_bin_dtype_boundaries_and_64bit_catalog(tmp_path):
    assert _bin_dtype(256) == np.dtype('uint8')
    assert _bin_dtype(257) == np.dtype('uint16')
    assert _bin_dtype(65537) == np.dtype('uint32')
    assert _bin_dtype(2**32 + 1) == np.dtype('uint64')
    # Catalog arithmetic uses Python/int64, never a uint32 spike permutation.
    offsets = np.cumsum(np.array([2**32 + 7, 5], dtype=np.int64))
    assert int(offsets[-1]) == 2**32 + 12


def test_fortran_positions_preserve_declared_depth_axis(tmp_path):
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.tests.test_source import make_sorting

    params = make_sorting(tmp_path / 'source')
    (params.parent / 'spike_depths.npy').unlink()
    params.write_text(params.read_text() + 'depth_axis = 1\n')
    positions = np.asfortranarray(np.column_stack((np.arange(12) + 1000, np.arange(12) * 7)))
    np.save(params.parent / 'spike_positions.npy', positions)
    bounded = prepare_large(params, tmp_path / 'bounded', bin_seconds=5)
    build_summary_store(bounded, tmp_path / 'store', batch_size=3)
    assignments = np.load(params.parent / 'spike_clusters.npy')
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        for unit in engine.manifest['unit_ids']:
            data = np.concatenate(list(store.batches(unit, 2)))
            np.testing.assert_array_equal(data['depth'], positions[assignments == unit, 1])


def test_attach_completed_store_after_reference_loss(tmp_path):
    from phy.apps.large_recording.summary_store import attach_summary_store

    _, bounded = package(tmp_path)
    output = build_summary_store(bounded, tmp_path / 'store')
    reference = bounded / '.phy' / 'live-summary-store.json'
    reference.unlink()
    attach_summary_store(bounded, output)
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        assert len(np.concatenate(list(store.batches(0, 2)))) == 4


def test_real_sparse_payload_seek_beyond_32bit_rows(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from phy.apps.large_recording.source import hash_json
    from phy.apps.large_recording.summary_store import binding

    count = 2**32 + 7
    engine = SimpleNamespace(
        path=tmp_path / 'bounded',
        binding={'test': 'large offsets'},
        bins=4,
        fields=['amplitude'],
        manifest={
            'unit_ids': [0, 1],
            'counts': [count, 1],
            'source_array_inventory': {'amplitudes.npy': {'dtype': 'float32'}},
        },
    )
    store_path = tmp_path / 'store'
    store_path.mkdir()
    dtype = np.dtype([('bin', 'uint8'), ('amplitude', 'float32')])
    row = np.array([(2, 3.5)], dtype=dtype).tobytes()
    with (store_path / 'data.bin').open('wb') as stream:
        import os

        if os.name == 'nt':
            import ctypes
            import msvcrt

            # NTFS requires explicit sparse marking before extending a file:
            # https://learn.microsoft.com/windows/win32/api/winioctl/ni-winioctl-fsctl_set_sparse
            returned = ctypes.c_ulong()
            handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
            success = ctypes.windll.kernel32.DeviceIoControl(
                handle, 0x900C4, None, 0, None, 0, ctypes.byref(returned), None
            )
            if not success:
                pytest.skip('Test filesystem cannot create sparse files.')
        stream.seek(count * dtype.itemsize)
        stream.write(row)
    index = {
        'binding': binding(engine),
        'dtype': dtype.descr,
        'payload_bytes': (count + 1) * dtype.itemsize,
        'units': {
            '0': {'start': 0, 'count': count, 'sha256': 'unused'},
            '1': {'start': count, 'count': 1, 'sha256': hashlib.sha256(row).hexdigest()},
        },
    }
    (store_path / 'index.json').write_text(json.dumps(index))
    (engine.path / '.phy').mkdir(parents=True)
    (engine.path / '.phy' / 'live-summary-store.json').write_text(
        json.dumps(
            {
                'path': str(store_path),
                'index_sha256': hash_json(index),
            }
        )
    )
    with SummaryStore(engine) as store:
        data = np.concatenate(list(store.batches(1, 2)))
    assert data['bin'].tolist() == [2]
    assert data['amplitude'].tolist() == [3.5]


@pytest.mark.parametrize('operator', ['<', '<=', '>', '>='])
def test_indexed_nested_histories_match_independent_replay_offline(
    tmp_path, monkeypatch, operator
):
    from phy.apps.large_recording.live_summaries import membership_expressions
    from phy.apps.large_recording.tests.test_live_summaries import merge, oracle, split

    source, bounded = package(tmp_path, columns=True, positions=True)
    build_summary_store(bounded, tmp_path / 'store', batch_size=3)
    events = [split(1, [100, 101]), merge([101, 2], 102), split(102, [103, 104], threshold=7)]
    events[-1]['predicate']['operator'] = operator
    expressions = membership_expressions([0, 1, 2, 99], events)
    with LiveSummaryEngine(bounded) as engine:
        expected = {u: oracle(source, engine, events, u) for u in expressions}
    source.rename(tmp_path / 'source-offline')

    def no_source(*args, **kwargs):
        raise AssertionError('Indexed summaries opened the original source')

    monkeypatch.setattr(LiveSummaryEngine, '_open', no_source)
    progress = []
    with LiveSummaryEngine(bounded, batch_size=2) as engine:
        results = engine.resolve(expressions, progress=lambda *p: progress.append(p))
        for unit, summary in results.items():
            for name, values in engine._flat(summary).items():
                np.testing.assert_allclose(values, engine._flat(expected[unit])[name])
    # Only original units 1 and 2 are read, once each, for all their descendants.
    assert progress[-1] == ('reading prepared unit summaries', 6, 6)


def test_corrupt_index_falls_back_without_publishing_partial_results(tmp_path, monkeypatch):
    from phy.apps.large_recording.live_summaries import membership_expressions
    from phy.apps.large_recording.tests.test_live_summaries import oracle, split

    source, bounded = package(tmp_path)
    output = build_summary_store(bounded, tmp_path / 'store')
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        offset = store.index['units']['0']['start'] * store.dtype.itemsize
    with (output / 'data.bin').open('r+b') as stream:
        stream.seek(offset + 1)
        stream.write(b'\xff')
    events = [split(0, [100, 101])]
    expressions = membership_expressions([0, 1, 2, 99], events)
    calls = []
    original = LiveSummaryEngine._open

    def opened(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LiveSummaryEngine, '_open', opened)
    with LiveSummaryEngine(bounded) as engine:
        results = engine.resolve(expressions)
        for unit in (100, 101):
            for name, values in engine._flat(results[unit]).items():
                np.testing.assert_allclose(
                    values, engine._flat(oracle(source, engine, events, unit))[name]
                )
    assert calls == [True]


def test_indexed_nonfinite_reject_is_semantic_error_not_source_fallback(tmp_path, monkeypatch):
    from phy.apps.large_recording.live_summaries import membership_expressions
    from phy.apps.large_recording.tests.test_live_summaries import split

    _, bounded = package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store')

    def no_source(*args, **kwargs):
        raise AssertionError('Semantic rejection should not rescan the original source')

    monkeypatch.setattr(LiveSummaryEngine, '_open', no_source)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101], nonfinite='reject')])
    with LiveSummaryEngine(bounded) as engine, pytest.raises(ValueError, match='Nonfinite'):
        engine.resolve(expressions)
    assert not list((bounded / '.phy' / 'live-summary-cache').glob('*.npz'))


def test_template_predicate_uses_validated_source_fallback(tmp_path):
    from phy.apps.large_recording.live_summaries import membership_expressions
    from phy.apps.large_recording.tests.test_live_summaries import oracle, split

    source, bounded = package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store')
    events = [split(0, [100, 101], attribute='spike_templates.npy', threshold=1)]
    expressions = membership_expressions([0, 1, 2, 99], events)
    progress = []
    with LiveSummaryEngine(bounded) as engine:
        results = engine.resolve(expressions, progress=lambda *p: progress.append(p))
        np.testing.assert_array_equal(
            results[100]['counts'], oracle(source, engine, events, 100)['counts']
        )
    assert progress[-1][0] == 'scanning full source'


def test_cancel_indexed_scan_never_publishes(tmp_path):
    from phy.apps.large_recording.live_summaries import membership_expressions
    from phy.apps.large_recording.tests.test_live_summaries import split

    _, bounded = package(tmp_path)
    build_summary_store(bounded, tmp_path / 'store')
    expressions = membership_expressions([0, 1, 2, 99], [split(0, [100, 101])])
    stopped = [False]

    def progress(*args):
        stopped[0] = True

    with LiveSummaryEngine(bounded, batch_size=1) as engine, pytest.raises(CancelledError):
        engine.resolve(expressions, cancel=lambda: stopped[0], progress=progress)
    assert not list((bounded / '.phy' / 'live-summary-cache').glob('*.npz'))


def test_racing_publication_never_replaces_empty_target(tmp_path, monkeypatch):
    from phy.apps.large_recording import summary_store

    _, bounded = package(tmp_path)
    output = tmp_path / 'store'
    original = summary_store.publish_directory

    def race(staged, destination):
        destination.mkdir()
        return original(staged, destination)

    monkeypatch.setattr(summary_store, 'publish_directory', race)
    with pytest.raises(OSError):
        build_summary_store(bounded, output)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob('.*.building-*'))
    assert not (bounded / '.phy' / 'live-summary-store.json').exists()


def test_disk_failure_cleans_unpublished_store(tmp_path, monkeypatch):
    import errno

    from phy.apps.large_recording import summary_store

    _, bounded = package(tmp_path)

    def full(*args):
        raise OSError(errno.ENOSPC, 'disk full')

    monkeypatch.setattr(summary_store.os, 'fsync', full)
    with pytest.raises(OSError, match='disk full'):
        build_summary_store(bounded, tmp_path / 'store')
    assert not (tmp_path / 'store').exists()
    assert not list(tmp_path.glob('.*.building-*'))


def test_completed_store_can_attach_after_reference_write_failure(tmp_path, monkeypatch):
    from phy.apps.large_recording import summary_store

    _, bounded = package(tmp_path)
    output = tmp_path / 'store'
    original = summary_store._atomic_json

    def fail_reference(path, value):
        if path.name == summary_store.REFERENCE:
            raise OSError('reference write failed')
        original(path, value)

    monkeypatch.setattr(summary_store, '_atomic_json', fail_reference)
    with pytest.raises(OSError, match='reference write failed'):
        build_summary_store(bounded, output)
    assert output.is_dir()
    monkeypatch.setattr(summary_store, '_atomic_json', original)
    summary_store.attach_summary_store(bounded, output)
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        assert len(np.concatenate(list(store.batches(0, 2)))) == 4


def test_wide_real_scalars_use_float64_accumulators(tmp_path):
    from phy.apps.large_recording.live_summaries import membership_expressions
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.tests.test_live_summaries import split
    from phy.apps.large_recording.tests.test_source import make_sorting

    params = make_sorting(tmp_path / 'source')
    for name in ('amplitudes.npy', 'spike_depths.npy'):
        np.save(params.parent / name, np.load(params.parent / name).astype(np.longdouble))
    bounded = prepare_large(params, tmp_path / 'bounded', bin_seconds=5)
    expressions = membership_expressions([0, 1, 2, 99], [split(0, [100, 101])])
    with LiveSummaryEngine(bounded) as engine:
        baseline = engine.resolve(expressions)
    build_summary_store(bounded, tmp_path / 'store')
    import shutil

    shutil.rmtree(bounded / '.phy' / 'live-summary-cache')
    with LiveSummaryEngine(bounded) as engine:
        indexed = engine.resolve(expressions)
        for unit in expressions:
            for name, actual in engine._flat(indexed[unit]).items():
                np.testing.assert_array_equal(actual, engine._flat(baseline[unit])[name])


def test_post_publication_sync_failure_preserves_attachable_store(tmp_path, monkeypatch):
    from phy.apps.large_recording import summary_store

    _, bounded = package(tmp_path)
    output = tmp_path / 'store'
    original = summary_store._fsync_directory

    def fail_parent(path):
        if path == output.parent and output.exists():
            raise OSError('parent sync failed')
        original(path)

    monkeypatch.setattr(summary_store, '_fsync_directory', fail_parent)
    with pytest.raises(OSError, match='parent sync failed'):
        build_summary_store(bounded, output)
    assert output.is_dir()
    assert not list(tmp_path.glob('.*.building-*'))
    monkeypatch.setattr(summary_store, '_fsync_directory', original)
    summary_store.attach_summary_store(bounded, output)


def test_automatic_build_attachment_preserves_newer_store_choice(tmp_path):
    from phy.apps.large_recording.summary_store import (
        SummaryStorePublishedError,
        attach_summary_store,
        summary_store_reference,
    )

    _, bounded = package(tmp_path)
    first = build_summary_store(bounded, tmp_path / 'first', attach=False)

    def choose_first():
        attach_summary_store(bounded, first)

    with pytest.raises(SummaryStorePublishedError, match='changed while'):
        build_summary_store(bounded, tmp_path / 'second', before_publish=choose_first)
    assert (tmp_path / 'second' / 'index.json').is_file()
    reference = json.loads(summary_store_reference(bounded))
    assert reference['path'] == str(first)
