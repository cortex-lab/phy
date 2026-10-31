"""Exact source clock, alignment, and identity regressions."""

import json

import numpy as np
import pytest

from phy.apps.large_recording.source import Source, hash_file


def make_sorting(path, columns=False):
    path.mkdir()
    samples = np.array([0, 1, 1, 2, 5, 6, 10, 11, 14, 18, 19, 19], dtype=np.int64)
    clusters = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 99, 99], dtype=np.int32)
    templates = np.arange(len(samples), dtype=np.int32) % 3
    arrays = {
        'spike_times.npy': samples,
        'spike_clusters.npy': clusters,
        'spike_templates.npy': templates,
        'amplitudes.npy': np.arange(len(samples), dtype=np.float32),
        'spike_positions.npy': np.column_stack((np.zeros(len(samples)), samples * 2)),
        'spike_depths.npy': samples.astype(np.float64) * 2,
        'templates.npy': np.arange(3 * 8 * 3, dtype=np.float32).reshape(3, 8, 3),
        'channel_map.npy': np.array([2, 0, 1], dtype=np.int32),
        'channel_positions.npy': np.array([[0, 0], [0, 10], [0, 20]]),
        'pc_features.npy': np.arange(len(samples) * 4, dtype=np.float32).reshape(-1, 2, 2),
        'pc_feature_ind.npy': np.array([[0, 2], [2, 1], [1, 0]], dtype=np.int32),
    }
    for name, array in arrays.items():
        if columns and name in ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy'):
            array = array[:, None]
        np.save(path / name, array)
    ids = np.array([0, 2, 3, 8, 11], dtype=np.int64)
    channels = np.array([[0, -1], [2, 1], [1, -1], [2, -1], [0, 1]], dtype=np.int32)
    waveforms = np.zeros((len(ids), 8, 2), dtype=np.float32)
    for row, spike in enumerate(ids):
        for slot, channel in enumerate(channels[row]):
            if channel != -1:
                waveforms[row, :, slot] = spike + channel + np.arange(8) / 10
    for suffix, array in (('spikes', ids), ('channels', channels), ('waveforms', waveforms)):
        np.save(path / f'_phy_spikes_subset.{suffix}.npy', array)
    (path / 'params.py').write_text(
        "sample_rate = 1.0\nnum_samples = 20\ndat_path = None\nn_channels_dat = 3\ndtype = 'int16'\n"
    )
    (path / 'cluster_group.tsv').write_text(
        'cluster_id\tgroup\n0\tgood\n1\tmua\n2\tgood\n99\tnoise\n'
    )
    return path / 'params.py'


@pytest.mark.parametrize('columns', [False, True])
def test_source_batches_ties_counts_columns(tmp_path, columns):
    source = Source(make_sorting(tmp_path / 'full', columns), batch_size=2)
    assert source.n_spikes == 12
    assert source.unit_ids.tolist() == [0, 1, 2, 99]
    assert source.counts.tolist() == [4, 3, 3, 2]
    assert all(batch['spike_times.npy'].ndim == 1 for batch in source.iter_batches())
    assert all(isinstance(array, np.memmap) for array in source.arrays.values())
    assert source.metadata['group'][99] == 'noise'


def test_declared_waveform_source_identity_is_checked(tmp_path):
    params = make_sorting(tmp_path / 'full')
    manifest = {
        'format': 'phy_spikes_subset',
        'format_version': 1,
        'recording': {'num_samples': 20, 'sampling_frequency_hz': 1},
        'source_sorting': {'num_spikes': 12, 'spike_times_sha256': '0' * 64},
    }
    path = params.parent / '_phy_spikes_subset.manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fingerprint'):
        Source(params)
    manifest['source_sorting']['spike_times_sha256'] = hash_file(params.parent / 'spike_times.npy')
    path.write_text(json.dumps(manifest))
    Source(params).close()
    manifest['source_sorting']['num_spikes'] = 13
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='spike count'):
        Source(params)


def test_failed_source_validation_closes_readers_and_rejects_unsupported_ids(
    tmp_path, monkeypatch
):
    params = make_sorting(tmp_path / 'full')
    labels = np.load(params.parent / 'spike_clusters.npy').astype(np.int64)
    labels[-1] = 2**31
    np.save(params.parent / 'spike_clusters.npy', labels)
    original = np.load
    readers = []

    def read(*args, **kwargs):
        array = original(*args, **kwargs)
        readers.append(array)
        return array

    monkeypatch.setattr(np, 'load', read)
    with pytest.raises(ValueError, match='int32'):
        Source(params)
    assert all(array._mmap.closed for array in readers)


def test_identity_assignment_and_feature_layers(tmp_path):
    params = make_sorting(tmp_path / 'full')
    source = Source(params)
    source.close()
    clusters = np.load(params.parent / 'spike_clusters.npy')
    clusters[clusters == 1] = 0
    np.save(params.parent / 'spike_clusters.npy', clusters)
    merged = Source(params)
    merged.close()
    assert merged.identity == source.identity
    assert merged.revision != source.revision
    values = np.load(params.parent / 'amplitudes.npy')
    values[0] = 99
    np.save(params.parent / 'amplitudes.npy', values)
    altered = Source(params)
    altered.close()
    assert altered.identity != merged.identity
    assert altered.revision == merged.revision
    assert (
        hash_file(params.parent / 'amplitudes.npy') == altered.identity['arrays']['amplitudes.npy']
    )


def test_exposure_gaps_and_missing_duration(tmp_path):
    params = make_sorting(tmp_path / 'full')
    source = Source(params, valid_intervals=[[0, 7], [10, 20]])
    assert source.valid_intervals.tolist() == [[0, 7], [10, 20]]
    with pytest.raises(ValueError, match='outside valid'):
        Source(params, valid_intervals=[[0, 5], [10, 20]])
    params.write_text(params.read_text().replace('num_samples = 20\n', ''))
    with pytest.raises(ValueError, match='exposure is missing'):
        Source(params)
    assert Source(params, duration_seconds=20).num_samples == 20
    with pytest.raises(ValueError, match='integer sample count'):
        Source(params, duration_seconds=20.5)


@pytest.mark.parametrize(
    'failure', ['order', 'alignment', 'unknown', 'segment', 'waveforms', 'ids']
)
def test_reject_ambiguous_or_misaligned_sources(tmp_path, failure):
    params = make_sorting(tmp_path / 'full')
    path = params.parent
    if failure == 'order':
        np.save(path / 'spike_times.npy', np.arange(12)[::-1])
    elif failure == 'alignment':
        np.save(path / 'pc_features.npy', np.zeros((11, 2, 2)))
    elif failure == 'unknown':
        np.save(path / 'mysterious.npy', np.ones(12))
    elif failure == 'segment':
        params.write_text(params.read_text() + 'num_segments = 2\n')
    elif failure == 'waveforms':
        (path / '_phy_spikes_subset.channels.npy').unlink()
    else:
        np.save(path / '_phy_spikes_subset.spikes.npy', np.array([0, 2, 2, 8, 11], dtype=np.int64))
    with pytest.raises(ValueError):
        Source(params, batch_size=2)


def test_metadata_json_preserves_typed_values(tmp_path):
    params = make_sorting(tmp_path / 'full')
    (params.parent / 'cluster_group.tsv').write_text('cluster_id\tgroup\n0\tgood\n')
    metadata = {'group': {'0': 'good'}, 'custom': {'99': None, '2': 4}}
    (params.parent / '_phy_cluster_metadata.json').write_text(json.dumps(metadata))
    assert Source(params).metadata == {'group': {0: 'good'}, 'custom': {99: None, 2: 4}}


def test_replay_lineage_retains_zero_count_catalog_and_allocator(tmp_path):
    params = make_sorting(tmp_path / 'full')
    source = Source(params)
    lineage = {
        'format': 'phy_curation_lineage',
        'format_version': 1,
        'coordinate_identity': source.identity,
        'revision': source.revision,
        'unit_ids': [0, 1, 2, 99, 103],
        'next_cluster_id': 105,
    }
    (params.parent / '_phy_curation_lineage.json').write_text(json.dumps(lineage))
    reopened = Source(params)
    assert reopened.next_cluster_id == 105
    assert reopened.unit_ids[-1] == 103
    assert reopened.counts[-1] == 0
    assert reopened.identity == source.identity
    lineage['next_cluster_id'] = 103
    (params.parent / '_phy_curation_lineage.json').write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match='collides'):
        Source(params)


def test_unknown_upstream_manifest_version_is_rejected(tmp_path):
    params = make_sorting(tmp_path / 'full')
    (params.parent / '_phy_spikes_subset.manifest.json').write_text(
        json.dumps({'format': 'phy_spikes_subset', 'format_version': 2})
    )
    with pytest.raises(ValueError, match='Unsupported waveform manifest'):
        Source(params)


def test_integer_clock_beyond_32_bits(tmp_path):
    from phy.apps.tests.large_recording_fixtures import make_long_recording_fixture

    fixture = make_long_recording_fixture(tmp_path)
    source = Source(fixture.full / 'params.py', batch_size=3)
    assert source.num_samples == 1000 * 3600 * 30000
    assert source.arrays['spike_times.npy'][-1] > np.iinfo(np.int32).max
    assert source.n_spikes == 32


def test_source_heap_does_not_scale_with_spike_arrays(tmp_path):
    import tracemalloc

    peaks = []
    for spikes in (1000, 250000):
        path = tmp_path / str(spikes)
        path.mkdir()
        for name, dtype in (
            ('spike_times.npy', np.int64),
            ('spike_clusters.npy', np.int32),
            ('spike_templates.npy', np.int32),
        ):
            values = np.lib.format.open_memmap(
                path / name, mode='w+', dtype=dtype, shape=(spikes,)
            )
            for start in range(0, spikes, 1024):
                stop = min(start + 1024, spikes)
                values[start:stop] = np.arange(start, stop) if name == 'spike_times.npy' else 0
            values.flush()
            del values
        (path / 'params.py').write_text(f'sample_rate = 1\nnum_samples = {spikes}\n')
        tracemalloc.start()
        source = Source(path / 'params.py', batch_size=1024)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert source.n_spikes == spikes
        assert sum(source.counts) == spikes
        peaks.append(peak)
    # This checks allocated heap; file-backed OS pages are intentionally separate.
    assert peaks[-1] < 4 * 1024 * 1024
    assert peaks[-1] - peaks[0] < 3 * 1024 * 1024


def test_explicit_intervals_cannot_override_changed_producer_exposure(tmp_path):
    params = make_sorting(tmp_path / 'full')
    params.write_text(params.read_text() + 'valid_intervals = [[0, 7], [10, 20]]\n')
    with pytest.raises(ValueError, match='disagree with acquisition metadata'):
        Source(params, valid_intervals=[[0, 20]])
    with Source(params) as source:
        arrays = list(source.arrays.values())
        assert source.valid_intervals.tolist() == [[0, 7], [10, 20]]
    assert not source.arrays
    assert all(array._mmap.closed for array in arrays)


def test_exporter_channel_sidecars_and_raw_map_validation(tmp_path):
    params = make_sorting(tmp_path / 'full')
    np.save(
        params.parent / 'channel_map_si.npy', np.array(['hardware2', 'hardware0', 'hardware1'])
    )
    np.save(params.parent / 'channel_groups.npy', np.array([1, 1, 0]))
    with Source(params) as source:
        assert 'channel_map_si.npy' in source.arrays
    params.write_text(params.read_text().replace('n_channels_dat = 3', 'n_channels_dat = 2'))
    with pytest.raises(ValueError, match='outside n_channels_dat'):
        Source(params)


@pytest.mark.parametrize(
    'positions', [np.zeros((3, 1)), np.zeros((3, 3)), np.full((3, 2), np.nan)]
)
def test_channel_geometry_must_match_gui_contract(tmp_path, positions):
    params = make_sorting(tmp_path / 'source')
    np.save(params.parent / 'channel_positions.npy', positions)
    with pytest.raises(ValueError, match='finite x/y'):
        Source(params)


def test_exclusive_publish_preserves_racing_empty_destination(tmp_path):
    from phy.apps.large_recording.source import publish_directory

    staged = tmp_path / 'stage'
    staged.mkdir()
    (staged / 'payload').write_text('complete')
    destination = tmp_path / 'destination'
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(staged, destination)
    assert destination.is_dir() and not list(destination.iterdir())
    assert (staged / 'payload').read_text() == 'complete'
    destination.rmdir()
    publish_directory(staged, destination)
    assert not staged.exists()
    assert (destination / 'payload').read_text() == 'complete'


def test_clock_endpoint_mismatch_fails_before_full_scan_or_hash(tmp_path, monkeypatch):
    from phy.apps.large_recording import source as module

    params = make_sorting(tmp_path / 'source')
    times = np.load(params.parent / 'spike_times.npy')
    times[-1] = 100
    np.save(params.parent / 'spike_times.npy', times)

    def unexpected(*args, **kwargs):
        pytest.fail('Clock endpoint mismatch must fail before scans or hashing.')

    monkeypatch.setattr(Source, 'iter_batches', unexpected)
    monkeypatch.setattr(module, 'hash_file', unexpected)
    with pytest.raises(ValueError, match='same time base'):
        Source(params)


def clock_manifest(params, recording):
    path = params.parent / '_phy_spikes_subset.manifest.json'
    path.write_text(
        json.dumps(
            {
                'format': 'phy_spikes_subset',
                'format_version': 1,
                'recording': recording,
            }
        )
    )
    return path


def test_explicit_clocks_and_legacy_identity(tmp_path):
    params = make_sorting(tmp_path / 'source')
    with Source(params) as source:
        legacy = source.identity
        assert source.time_base is None
    # Existing exports preserve their exact identity schema.
    assert set(legacy['time_base']) == {'sample_rate', 'num_samples', 'valid_intervals'}
    contract = {'kind': 'acquisition_samples', 'ticks_per_second': 1}
    params.write_text(params.read_text() + f'time_base = {contract!r}\n')
    clock_manifest(params, {'time_base': contract})
    with Source(params) as source:
        assert source.time_base == contract
        assert source.identity != legacy
        explicit = source.identity
    timestamps = params.parent / 'timestamps.json'
    timestamps.write_text('{"mapping": "producer-owned"}')
    with Source(params) as source:
        assert source.identity != explicit
        assert source.identity['time_base']['timestamps_sha256'] == hash_file(timestamps)
    timestamps.write_text('{"mapping": "changed"}')
    with Source(params) as source:
        assert source.identity['time_base']['timestamps_sha256'] == hash_file(timestamps)


@pytest.mark.parametrize(
    'contract',
    [
        None,
        {},
        {'kind': 'clock', 'ticks_per_second': 1},
        {'kind': 'elapsed_ticks', 'ticks_per_second': 0},
        {'kind': 'elapsed_ticks', 'ticks_per_second': float('inf')},
        {'kind': 'elapsed_ticks', 'ticks_per_second': True},
        {'kind': 'elapsed_ticks', 'ticks_per_second': '1'},
    ],
)
def test_invalid_explicit_clock_contract(tmp_path, contract):
    params = make_sorting(tmp_path / 'source')
    clock_manifest(params, {'time_base': contract})
    with pytest.raises(ValueError, match='time_base'):
        Source(params)


def test_timestamps_without_contract_fail_before_arrays(tmp_path, monkeypatch):
    params = make_sorting(tmp_path / 'source')
    (params.parent / 'timestamps.json').write_text('{}')
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('Unexpected array read'))
    with pytest.raises(ValueError, match='explicit time_base'):
        Source(params)


@pytest.mark.parametrize(
    'failure',
    [
        'clock',
        'ticks',
        'waveform_rate',
        'manifest_rate',
        'frame_coordinates',
        'missing_intervals',
        'raw',
        'window_coordinates',
        'sample_coordinates',
    ],
)
def test_explicit_elapsed_clock_rejects_ambiguous_producer_metadata(tmp_path, failure):
    params = make_sorting(tmp_path / 'source')
    contract = {'kind': 'elapsed_ticks', 'ticks_per_second': 1}
    recording = {'time_base': contract, 'valid_intervals': [[0, 20]]}
    if failure == 'clock':
        params.write_text(
            params.read_text()
            + "time_base = {'kind': 'acquisition_samples', 'ticks_per_second': 1}\n"
        )
    elif failure == 'ticks':
        recording['time_base'] = {**contract, 'ticks_per_second': 2}
    elif failure == 'waveform_rate':
        params.write_text(params.read_text() + 'waveform_sample_rate = 2\n')
    elif failure == 'manifest_rate':
        recording['sampling_frequency_hz'] = 2
    elif failure == 'frame_coordinates':
        recording['frame_coordinates'] = 'before any clock-time conversion'
    elif failure == 'missing_intervals':
        recording.pop('valid_intervals')
    elif failure == 'raw':
        params.write_text(params.read_text() + "dat_path = ['None', 'raw.bin']\n")
    elif failure == 'window_coordinates':
        np.save(params.parent / '_phy_spikes_subset.window_bounds.npy', [[0, 20]])
    else:
        np.save(params.parent / '_phy_spikes_subset.sample_index.npy', [0, 1, 2, 14, 19])
        recording['sample_index_coordinates'] = 'acquisition_samples'
    clock_manifest(params, recording)
    with pytest.raises(ValueError):
        Source(params)


@pytest.mark.parametrize('raw', [None, 'None', '', [], [None, 'None', '']])
def test_elapsed_clock_canonical_sidecars_and_no_raw_sentinels(tmp_path, raw):
    params = make_sorting(tmp_path / 'source')
    contract = {'kind': 'elapsed_ticks', 'ticks_per_second': 1}
    params.write_text(params.read_text() + f'dat_path = {raw!r}\n')
    recording = {
        'time_base': contract,
        'valid_intervals': [[0, 7], [10, 20]],
        'frame_coordinates': 'elapsed_ticks',
        'window_bounds_coordinates': 'elapsed_ticks',
        'sample_index_coordinates': 'elapsed_ticks',
    }
    clock_manifest(params, recording)
    np.save(params.parent / '_phy_spikes_subset.window_bounds.npy', [[0, 7], [10, 20]])
    np.save(params.parent / '_phy_spikes_subset.sample_index.npy', [0, 1, 2, 14, 19])
    with Source(params) as source:
        assert source.time_base == contract
        assert source.upstream_recording == recording
        assert source.valid_intervals.tolist() == [[0, 7], [10, 20]]


@pytest.mark.parametrize(
    'values',
    [
        [0, 1, 2, 14],
        [[0], [1], [2], [14], [19]],
        [0.0, 1.0, 2.0, 14.0, 19.0],
        [-1, 1, 2, 14, 19],
        [0, 1, 2, 14, 20],
        [0, 1, 2, 13, 19],
    ],
)
def test_waveform_sample_index_requires_exact_canonical_alignment(tmp_path, values):
    params = make_sorting(tmp_path / 'source')
    np.save(params.parent / '_phy_spikes_subset.sample_index.npy', values)
    with pytest.raises(ValueError, match='sample_index'):
        Source(params)


def test_waveform_sample_index_legacy_default_is_acquisition_samples(tmp_path):
    params = make_sorting(tmp_path / 'source', columns=True)
    np.save(params.parent / '_phy_spikes_subset.sample_index.npy', [0, 1, 2, 14, 19])
    with Source(params) as source:
        assert '_phy_spikes_subset.sample_index.npy' in source.identity['arrays']
