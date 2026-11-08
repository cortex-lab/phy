"""Small, deterministic on-disk example of a 1000-hour two-stage Phy dataset.

This is reference test data, not a prepare-large implementation. The fake
recording has 32 spikes spread across 1000 hours, not 1000 hours of raw traces.
The full and bounded folders use ordinary Phy filenames.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LongRecordingFixture:
    full: Path
    bounded: Path
    global_ids: np.ndarray
    upstream_waveform_ids: np.ndarray
    bounded_waveform_ids: np.ndarray


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_remap(global_ids, upstream_ids):
    """Reference intersection for sorted, unique, nonnegative int64 row IDs."""
    global_ids = np.asarray(global_ids)
    upstream_ids = np.asarray(upstream_ids)
    for ids in (global_ids, upstream_ids):
        if ids.ndim != 1 or ids.dtype != np.dtype('int64'):
            raise ValueError('Row IDs must be one-dimensional int64 arrays.')
        if np.any(ids < 0) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError('Row IDs must be nonnegative, sorted, and unique.')
    if not len(global_ids) or not len(upstream_ids):
        empty = np.array([], dtype=np.int64)
        return empty, empty.copy()
    positions = np.searchsorted(global_ids, upstream_ids)
    keep = positions < len(global_ids)
    keep[keep] &= global_ids[positions[keep]] == upstream_ids[keep]
    return positions[keep].astype(np.int64), np.flatnonzero(keep).astype(np.int64)


def make_long_recording_fixture(root):
    """Write one full Phy dataset and its expected bounded counterpart."""
    root = Path(root)
    full, bounded = root / 'full', root / 'bounded'
    full.mkdir(parents=True)
    bounded.mkdir(parents=True)

    fs = 30_000
    num_samples = 1000 * 3600 * fs  # Far beyond signed int32 sample indices.
    samples = np.linspace(100, num_samples - 100, 32, dtype=np.int64)
    samples[8] = samples[5]  # Simultaneous spikes retain distinct source row identities.
    samples[6:8] = samples[5]
    units = (np.arange(32) % 3).astype(np.int32)
    amplitudes = np.arange(1, 33, dtype=np.float32)
    templates = np.zeros((3, 12, 4), dtype=np.float32)
    for unit in range(3):
        templates[unit, :, unit] = -np.arange(12, dtype=np.float32)

    for name, array in (
        ('spike_times.npy', samples),
        ('spike_clusters.npy', units),
        ('spike_templates.npy', units),
        ('amplitudes.npy', amplitudes),
        ('templates.npy', templates),
        ('channel_map.npy', np.arange(4, dtype=np.int32)),
        ('channel_positions.npy', np.array([[0, 0], [20, 0], [0, 20], [20, 20]])),
    ):
        np.save(full / name, array)
    (full / 'params.py').write_text(
        "sample_rate = 30000.0\ndat_path = None\nn_channels_dat = 4\ndtype = 'int16'\n",
        encoding='utf-8',
    )

    upstream_ids = np.array([1, 2, 8, 13, 18, 21, 28, 31], dtype=np.int64)
    channels = np.array(
        [[0, 1, -1] if i % 2 == 0 else [2, 3, -1] for i in range(len(upstream_ids))],
        dtype=np.int32,
    )
    waveforms = np.zeros((len(upstream_ids), 12, 3), dtype=np.float32)
    for row, spike_id in enumerate(upstream_ids):
        waveforms[row, :, 0] = spike_id + np.arange(12) / 12
        waveforms[row, :, 1] = -(spike_id + np.arange(12) / 12)
    stem = '_phy_spikes_subset'
    np.save(full / f'{stem}.spikes.npy', upstream_ids)
    np.save(full / f'{stem}.channels.npy', channels)
    np.save(full / f'{stem}.waveforms.npy', waveforms)

    # Proposed, optional upstream window metadata: simple homogeneous arrays.
    starts = np.minimum(samples[upstream_ids] - 10, num_samples - 30 * fs)
    bounds = np.column_stack((starts, starts + 30 * fs)).astype(np.int64)
    types = np.zeros(len(upstream_ids), dtype=np.uint8)
    types[-1] = 1  # rescue
    strata = (samples[upstream_ids] // (3600 * fs)).astype(np.int32)
    strata[-1] = -1  # rescue has no survey stratum
    np.save(full / f'{stem}.window_bounds.npy', bounds)
    np.save(full / f'{stem}.window_types.npy', types)
    np.save(full / f'{stem}.window_strata.npy', strata)
    source_hash = _sha256(full / 'spike_times.npy')
    (full / f'{stem}.manifest.json').write_text(
        json.dumps(
            {
                'format': 'phy_spikes_subset',
                'format_version': 1,
                'source_sorting': {'num_spikes': len(samples), 'spike_times_sha256': source_hash},
                'recording': {
                    'sampling_frequency_hz': fs,
                    'num_samples': num_samples,
                    'num_channels': 4,
                    'num_segments': 1,
                },
                'selection': {'algorithm': 'synthetic_windows_v1', 'seed': 42},
                'waveform_extraction': {'nbefore_samples': 4, 'nafter_samples': 8},
                'window_types': {'0': 'survey', '1': 'rescue'},
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )

    global_ids = np.array([0, 2, 5, 8, 13, 21, 25, 31], dtype=np.int64)
    for name in ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy', 'amplitudes.npy'):
        np.save(bounded / name, np.load(full / name)[global_ids])
    for name in ('templates.npy', 'channel_map.npy', 'channel_positions.npy', 'params.py'):
        shutil.copyfile(full / name, bounded / name)
    np.save(bounded / '_phy_spike_global_ids.npy', global_ids)

    bounded_rows, waveform_rows = _reference_remap(global_ids, upstream_ids)
    np.save(bounded / f'{stem}.spikes.npy', bounded_rows)
    np.save(bounded / f'{stem}.channels.npy', channels[waveform_rows])
    np.save(bounded / f'{stem}.waveforms.npy', waveforms[waveform_rows])
    (bounded / '_phy_curation_manifest.json').write_text(
        json.dumps(
            {
                'format': 'phy_bounded_curation',
                'format_version': 1,
                'source': {'num_spikes': len(samples), 'spike_times_sha256': source_hash},
                'global_spike_ids_file': '_phy_spike_global_ids.npy',
                'duration_seconds': 1000 * 3600,
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    return LongRecordingFixture(full, bounded, global_ids, upstream_ids, bounded_rows)
