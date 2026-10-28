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
    with LiveSummaryEngine(bounded) as engine:
        with pytest.raises(ValueError, match='binding'):
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
