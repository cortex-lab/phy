"""Reference contract tests for future prepare-large and waveform remapping.

These tests deliberately do not exercise a production prepare-large command:
that command does not exist yet. Later implementations should reuse this
fixture and replace reference-only checks with command-level assertions.
"""

import hashlib
import json

import numpy as np
import pytest
from phylib.io.model import load_model

from .large_recording_fixtures import _reference_remap, make_long_recording_fixture


@pytest.fixture
def long_recording(tmp_path):
    return make_long_recording_fixture(tmp_path)


def test_long_recording_uses_exact_64_bit_sample_times(long_recording):
    full = np.load(long_recording.full / 'spike_times.npy')
    bounded = np.load(long_recording.bounded / 'spike_times.npy')
    assert full.dtype == bounded.dtype == np.dtype('int64')
    assert full[-1] > np.iinfo(np.int32).max
    assert full[-1] / 30_000 > 999 * 3600
    np.testing.assert_array_equal(bounded, full[long_recording.global_ids])
    assert np.all(np.diff(long_recording.global_ids) > 0)
    assert np.all(np.diff(bounded) >= 0)
    assert bounded[2] == bounded[3]  # equal times do not collapse distinct spikes


def test_two_stage_row_indices_and_waveform_data(long_recording):
    full = long_recording.full
    bounded = long_recording.bounded
    global_ids = np.load(bounded / '_phy_spike_global_ids.npy')
    full_waveform_ids = np.load(full / '_phy_spikes_subset.spikes.npy')
    bounded_waveform_ids = np.load(bounded / '_phy_spikes_subset.spikes.npy')
    assert global_ids.dtype == bounded_waveform_ids.dtype == np.dtype('int64')
    np.testing.assert_array_equal(
        global_ids[bounded_waveform_ids], np.intersect1d(full_waveform_ids, global_ids)
    )
    upstream_rows = np.searchsorted(full_waveform_ids, global_ids[bounded_waveform_ids])
    for suffix in ('channels', 'waveforms'):
        np.testing.assert_array_equal(
            np.load(bounded / f'_phy_spikes_subset.{suffix}.npy'),
            np.load(full / f'_phy_spikes_subset.{suffix}.npy')[upstream_rows],
        )
    # A bounded row identifies one source row, and never a waveform tensor row.
    assert bounded_waveform_ids.tolist() == [1, 3, 4, 5, 7]
    assert global_ids[bounded_waveform_ids].tolist() == [2, 8, 13, 21, 31]


def test_ordinary_phy_waveform_files_are_aligned(long_recording):
    bounded = long_recording.bounded
    spike_ids = np.load(bounded / '_phy_spikes_subset.spikes.npy')
    channels = np.load(bounded / '_phy_spikes_subset.channels.npy')
    waveforms = np.load(bounded / '_phy_spikes_subset.waveforms.npy', mmap_mode='r')
    assert spike_ids.ndim == 1 and np.all(np.diff(spike_ids) > 0)
    assert channels.dtype == np.dtype('int32')
    assert channels.shape == (len(spike_ids), waveforms.shape[2])
    assert waveforms.shape == (len(spike_ids), 12, 3)
    assert np.all((channels == -1) | ((channels >= 0) & (channels < 4)))
    assert np.all(waveforms[:, :, -1] == 0)  # -1 means a zero-padded channel
    assert spike_ids[-1] < len(np.load(bounded / 'spike_times.npy'))


def test_window_arrays_use_samples_and_simple_dtypes(long_recording):
    full = long_recording.full
    bounds = np.load(full / '_phy_spikes_subset.window_bounds.npy')
    types = np.load(full / '_phy_spikes_subset.window_types.npy')
    strata = np.load(full / '_phy_spikes_subset.window_strata.npy')
    spike_samples = np.load(full / 'spike_times.npy')
    assert bounds.shape == (len(types), 2)
    assert len(types) == len(strata)
    assert bounds.dtype == np.dtype('int64')
    assert types.dtype == np.dtype('uint8')
    assert strata.dtype == np.dtype('int32')
    assert np.all(bounds[:, 1] > bounds[:, 0])
    assert np.all(bounds[:, 1] <= 1000 * 3600 * 30_000)
    assert np.all((types == 0) | (types == 1))
    assert strata[-1] == -1 and types[-1] == 1
    source_waveform_times = spike_samples[long_recording.upstream_waveform_ids]
    assert np.all((bounds[:, 0] <= source_waveform_times) & (source_waveform_times < bounds[:, 1]))


def test_reference_remap_handles_empty_and_disjoint_inputs():
    a = np.array([14, 19, 117], dtype=np.int64)
    b = np.array([2, 4, 119], dtype=np.int64)
    for left, right in ((a, b), (a, b[:0]), (a[:0], b), (a[:0], b[:0])):
        bounded_rows, source_rows = _reference_remap(left, right)
        assert bounded_rows.dtype == source_rows.dtype == np.dtype('int64')
        assert bounded_rows.shape == source_rows.shape == (0,)
    rows, upstream_rows = _reference_remap(a, np.array([14, 15, 117, 118]))
    assert rows.tolist() == [0, 2]
    assert upstream_rows.tolist() == [0, 2]


@pytest.mark.parametrize('ids', [[2, 1], [1, 1], [-1, 1]])
def test_reference_remap_rejects_invalid_row_order(ids):
    valid = np.array([1, 2], dtype=np.int64)
    invalid = np.array(ids, dtype=np.int64)
    for left, right in ((valid, invalid), (invalid, valid)):
        with pytest.raises(ValueError, match='nonnegative, sorted, and unique'):
            _reference_remap(left, right)


def test_manifests_bind_source_index_space_not_mutable_clusters(long_recording):
    full = long_recording.full
    bounded = long_recording.bounded
    source_hash = hashlib.sha256((full / 'spike_times.npy').read_bytes()).hexdigest()
    upstream = json.loads((full / '_phy_spikes_subset.manifest.json').read_text())
    downstream = json.loads((bounded / '_phy_curation_manifest.json').read_text())
    assert upstream['source_sorting']['spike_times_sha256'] == source_hash
    assert downstream['source']['spike_times_sha256'] == source_hash
    assert upstream['source_sorting']['num_spikes'] == len(np.load(full / 'spike_times.npy'))
    assert downstream['global_spike_ids_file'] == '_phy_spike_global_ids.npy'
    assert downstream['duration_seconds'] == 1000 * 3600
    assert hashlib.sha256((bounded / 'spike_times.npy').read_bytes()).hexdigest() != source_hash


def test_bounded_dataset_loads_with_existing_phylib(long_recording):
    """The old Phy loader consumes bounded rows, not original global indices."""
    model = load_model(long_recording.bounded / 'params.py')
    try:
        np.testing.assert_array_equal(
            model.spike_waveforms.spike_ids, long_recording.bounded_waveform_ids
        )
        assert model.n_spikes == len(long_recording.global_ids)
        assert model.n_samples_waveforms == 12
        assert model.spike_times[-1] > 999 * 3600
        np.testing.assert_array_equal(
            model.get_waveforms(long_recording.bounded_waveform_ids, channel_ids=[0, 1, 2, 3]),
            _expected_dense_waveforms(long_recording.bounded),
        )
    finally:
        model.close()


def _expected_dense_waveforms(path):
    """Independent dense expansion of the fixture's sparse channel slots."""
    channels = np.load(path / '_phy_spikes_subset.channels.npy')
    waveforms = np.load(path / '_phy_spikes_subset.waveforms.npy')
    expected = np.zeros((len(channels), 12, 4), dtype=waveforms.dtype)
    for row, channel_row in enumerate(channels):
        for slot, channel in enumerate(channel_row):
            if channel >= 0:
                expected[row, :, channel] = waveforms[row, :, slot]
    return expected
