"""Production bounded preparation: budgets, rows, waveforms, and publication."""

import json

import numpy as np
import pytest
from phylib.io.model import load_model

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.source import PER_SPIKE, hash_file
from phy.apps.large_recording.tests.test_source import make_sorting


@pytest.mark.parametrize('channel_map', [[0, 1, 2], [2, 0, 1]])
def test_prepare_exact_arrays_waveforms_and_source_unchanged(tmp_path, channel_map):
    params = make_sorting(tmp_path / 'full', columns=True)
    np.save(params.parent / 'channel_map.npy', np.asarray(channel_map, dtype=np.int32))
    before = {p.name: hash_file(p) for p in params.parent.iterdir()}
    output = prepare_large(
        params,
        tmp_path / 'bounded',
        max_spikes=8,
        window_seconds=3,
        stratum_seconds=20,
        seed=0,
        bin_seconds=5,
        batch_size=2,
    )
    manifest = json.loads((output / '_phy_curation_manifest.json').read_text())
    windows = np.load(output / manifest['windows_file'])
    samples = np.load(params.parent / 'spike_times.npy').reshape(-1)
    expected_ids = np.array(
        [
            index
            for index, sample in enumerate(samples)
            if any(start <= sample < end for start, end in windows)
        ],
        dtype=np.int64,
    )
    global_ids = np.load(output / manifest['global_spike_ids_file'])
    np.testing.assert_array_equal(global_ids, expected_ids)
    assert len(global_ids) <= 8
    for name in PER_SPIKE:
        if (params.parent / name).exists():
            np.testing.assert_array_equal(
                np.load(output / name), np.load(params.parent / name)[expected_ids]
            )
    source_waveform_ids = np.load(params.parent / '_phy_spikes_subset.spikes.npy')
    expected_waveform_rows = [
        row for row, spike in enumerate(source_waveform_ids) if spike in expected_ids
    ]
    np.testing.assert_array_equal(
        global_ids[np.load(output / '_phy_spikes_subset.spikes.npy')],
        source_waveform_ids[expected_waveform_rows],
    )
    for suffix in ('channels', 'waveforms'):
        np.testing.assert_array_equal(
            np.load(output / f'_phy_spikes_subset.{suffix}.npy'),
            np.load(params.parent / f'_phy_spikes_subset.{suffix}.npy')[expected_waveform_rows],
        )
    assert before == {p.name: hash_file(p) for p in params.parent.iterdir()}
    assert 'dat_path = None' in (output / 'params.py').read_text()
    assert manifest['next_cluster_id'] == 100
    assert sum(manifest['counts']) == len(samples)
    assert sum(manifest['bounded_counts']) == len(global_ids)
    model = load_model(output / 'params.py')
    try:
        ids = np.load(output / '_phy_spikes_subset.spikes.npy')
        channels = np.load(output / '_phy_spikes_subset.channels.npy')
        sparse = np.load(output / '_phy_spikes_subset.waveforms.npy')
        expected = np.zeros((len(ids), 8, 3), dtype=np.float32)
        for row, slots in enumerate(channels):
            for slot, channel in enumerate(slots):
                if channel >= 0:
                    expected[row, :, channel] = sparse[row, :, slot]
        np.testing.assert_array_equal(model.get_waveforms(ids, channel_ids=[0, 1, 2]), expected)
    finally:
        model.close()


def test_deterministic_budgets_and_zero_evidence_catalog(tmp_path):
    params = make_sorting(tmp_path / 'full')
    outputs = [
        prepare_large(
            params,
            tmp_path / name,
            max_spikes=5,
            window_seconds=3,
            stratum_seconds=20,
            seed=0,
            bin_seconds=5,
            batch_size=2,
        )
        for name in ('one', 'two')
    ]
    manifests = [
        json.loads((path / '_phy_curation_manifest.json').read_text()) for path in outputs
    ]
    assert manifests[0] == manifests[1]
    assert manifests[0]['unit_ids'][-1] == 99
    assert manifests[0]['bounded_counts'][-1] == 0
    assert manifests[0]['omitted_windows']
    for path in outputs:
        ids = np.load(path / '_phy_spike_global_ids.npy')
        times = np.load(path / 'spike_times.npy')
        assert len(ids) <= 5
        # All three tied samples at 1 retain their separate source identities.
        assert np.count_nonzero(times == 1) == 2


def test_rejected_budget_or_interrupted_publication_leaves_no_output(tmp_path, monkeypatch):
    params = make_sorting(tmp_path / 'full')
    with pytest.raises(ValueError, match='No bounded evidence'):
        prepare_large(
            params, tmp_path / 'empty', max_spikes=1, window_seconds=3, stratum_seconds=20, seed=0
        )
    with pytest.raises(ValueError, match='Output budget'):
        prepare_large(params, tmp_path / 'too-big', max_bytes=10)
    import phy.apps.large_recording.prepare as module

    def fail(*args, **kwargs):
        raise OSError('simulated interruption')

    monkeypatch.setattr(module, 'build_summaries', fail)
    with pytest.raises(OSError, match='interruption'):
        prepare_large(params, tmp_path / 'interrupted', window_seconds=20, stratum_seconds=20)
    assert not (tmp_path / 'interrupted').exists()
    assert not list(tmp_path.glob('.interrupted.prepare-*'))


def test_prepare_missing_waveforms_is_explicit(tmp_path):
    params = make_sorting(tmp_path / 'full')
    for path in params.parent.glob('_phy_spikes_subset.*.npy'):
        path.unlink()
    output = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    manifest = json.loads((output / '_phy_curation_manifest.json').read_text())
    assert not manifest['capabilities']['stored_waveforms']
    assert not (output / '_phy_spikes_subset.waveforms.npy').exists()


def test_spikeinterface_singular_sparse_template_indices(tmp_path):
    params = make_sorting(tmp_path / 'full')
    templates = np.load(params.parent / 'templates.npy')[:, :, :2]
    templates[1, :, 1] = 0
    np.save(params.parent / 'templates.npy', templates)
    indices = np.array([[0, 2], [1, -1], [2, 0]], dtype=np.int32)
    np.save(params.parent / 'template_ind.npy', indices)
    output = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    model = load_model(output / 'params.py')
    try:
        np.testing.assert_array_equal(model.sparse_templates.cols, indices)
    finally:
        model.close()


def test_adjacent_acquisition_blocks_retain_observation_boundaries(tmp_path):
    params = make_sorting(tmp_path / 'full')
    output = prepare_large(
        params,
        tmp_path / 'bounded',
        valid_intervals=[[0, 10], [10, 20]],
        window_seconds=10,
        stratum_seconds=10,
    )
    manifest = json.loads((output / '_phy_curation_manifest.json').read_text())
    np.testing.assert_array_equal(np.load(output / manifest['windows_file']), [[0, 10], [10, 20]])
    assert manifest['windows'][0]['block'] != manifest['windows'][1]['block']


def test_disk_preflight_rejects_before_writing_package(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from phy.apps.large_recording import prepare

    params = make_sorting(tmp_path / 'full')
    monkeypatch.setattr(prepare.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='Insufficient disk'):
        prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    assert not (tmp_path / 'bounded').exists()
    assert not list(tmp_path.glob('.bounded.prepare-*'))


def test_final_windows_records_match_retained_samples_and_gaps(tmp_path):
    params = make_sorting(tmp_path / 'full')
    output = prepare_large(
        params,
        tmp_path / 'bounded',
        valid_intervals=[[0, 7], [10, 20]],
        window_seconds=5,
        stratum_seconds=20,
        bin_seconds=5,
    )
    manifest = json.loads((output / '_phy_curation_manifest.json').read_text())
    intervals = np.load(output / manifest['windows_file'])
    assert [[row['start'], row['end']] for row in manifest['windows']] == intervals.tolist()
    for row in manifest['windows']:
        start, end = row['start'], row['end']
        assert any(first <= start and end <= last for first, last in [[0, 7], [10, 20]])
        assert row['exposure_seconds'] == end - start
    assert manifest['capabilities']['predicate_split_attributes'] == ['amplitudes.npy']


def test_prepare_protects_loader_and_preserves_producer_documents(tmp_path):
    params = make_sorting(tmp_path / 'full')
    for name, value in (
        ('curation.json', {'merge': [[0, 1]]}),
        ('curation_review.json', {'accepted': []}),
    ):
        (params.parent / name).write_text(json.dumps(value))
    output = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    manifest = json.loads((output / '_phy_curation_manifest.json').read_text())
    assert manifest['bounded_files']['params.py'] == hash_file(output / 'params.py')
    assert manifest['bounded_files']['_phy_cluster_metadata.json'] == hash_file(
        output / '_phy_cluster_metadata.json'
    )
    for name in ('curation.json', 'curation_review.json'):
        assert (output / name).read_bytes() == (params.parent / name).read_bytes()
        assert name not in manifest['excluded_files']
