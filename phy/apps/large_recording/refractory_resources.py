"""Shared, allocation-based planning for exact RP counter builds."""

import json
from dataclasses import dataclass

import numpy as np

from .refractory import BIN_SECONDS, RefractoryUnavailable

DEFAULT_RP_MAX_BYTES = 67108864
DEFAULT_RP_MEMORY_BYTES = 67108864
# Authentication reads 1 MiB blocks; allow the old and new block to coexist.
AUTH_BYTES = 2 * 1048576
# Two NPY headers and ZIP entries fit within 1 KiB per unit. Catalog entries
# (IDs, filename, SHA256, status and scalar statistics) get another 1 KiB.
UNIT_ARCHIVE_BYTES = 1024
UNIT_CATALOG_BYTES = 1024
# Dictionaries, arrays, per-unit stream state and bounded partner tails.
UNIT_STATE_BYTES = 8192
# Includes pending input copies, dedup sorting/masks and feed temporaries.
COMPONENT_ROW_BYTES = 256


@dataclass(frozen=True)
class RefractoryBuildPlan:
    """Algorithm allocation allowances; not a ceiling on total process RSS."""

    units: int
    bins: int
    disk_bytes: int
    fixed_bytes: int
    component_bytes: int
    max_component: int
    batch_size: int
    row_bytes: int

    @property
    def memory_bytes(self):
        return self.fixed_bytes + self.component_bytes + self.batch_size * self.row_bytes


def estimate_refractory_disk(manifest, bins):
    """Bound the two int64 arrays, archive overhead and catalog, without curves."""
    units = len(manifest['unit_ids'])
    # Include arbitrary source binding/exposure metadata rather than assuming
    # a fixed number of acquisition intervals or a fixed source path length.
    binding_bytes = len(json.dumps(manifest.get('source', {})).encode())
    binding_bytes += len(json.dumps(manifest['valid_intervals']).encode())
    catalog_bytes = 65536 + binding_bytes + units * UNIT_CATALOG_BYTES
    if catalog_bytes > 16777216:
        raise RefractoryUnavailable('RP catalog exceeds reader resource limit.')
    return catalog_bytes + units * (16 * bins + UNIT_ARCHIVE_BYTES)


def plan_refractory_summaries(
    manifest,
    max_bytes=DEFAULT_RP_MAX_BYTES,
    memory_bytes=DEFAULT_RP_MEMORY_BYTES,
    batch_size=100000,
    native_grid_bytes=0,
    disk_option='--max-bytes',
    memory_option='--memory-bytes',
):
    """Validate geometry and budgets before allocating grids or scanning spikes.

    Counters and one shared exposure grid remain resident. Authentication and
    serialization scratch are reserved alongside them. Remaining memory is split
    between complete duplicate components and batch processing. A requested batch
    size is an upper bound, never an additional independent memory allowance.
    """
    if min(max_bytes, memory_bytes, batch_size) <= 0 or native_grid_bytes < 0:
        raise ValueError('RP budgets and batch size must be positive.')
    fs, samples = float(manifest['sample_rate']), int(manifest['num_samples'])
    if not np.isfinite(fs) or fs <= 0:
        raise RefractoryUnavailable('Unsupported RP sample rate.')
    width, radius, tr = round(BIN_SECONDS * fs), round(0.0003 * fs), round(0.001 * fs)
    if width <= 0 or tr <= radius:
        raise RefractoryUnavailable('Unsupported RP sample rate.')
    bins = (samples + width - 1) // width
    if samples <= 0 or samples > np.iinfo(np.int64).max - tr or bins > 200000:
        raise RefractoryUnavailable(
            'Scientific grid or sample arithmetic exceeds resource budget.'
        )
    units = len(manifest['unit_ids'])
    disk_bytes = estimate_refractory_disk(manifest, bins)
    geometry = f'{units} units × {bins} bins'
    if disk_bytes > max_bytes:
        raise ValueError(
            f'RP disk budget for {geometry}: need {disk_bytes} bytes, configured {max_bytes}; '
            f'increase {disk_option}.'
        )
    inventory = manifest.get('source_array_inventory', {})
    names = ('spike_times.npy', 'spike_clusters.npy', 'amplitudes.npy')
    input_bytes = sum(
        np.dtype(inventory.get(n, {}).get('dtype', 'float64')).itemsize for n in names
    )
    row_bytes = max(256, 128 + 8 * input_bytes)
    # np.savez uses at most a 16 MiB write buffer, or one counter array if smaller.
    serialization_bytes = min(16 * 1048576, bins * 8) + 65536
    fixed_bytes = (
        units * (16 * bins + UNIT_STATE_BYTES)
        + bins * 8
        + native_grid_bytes
        + AUTH_BYTES
        + serialization_bytes
    )
    minimum = fixed_bytes + units * COMPONENT_ROW_BYTES + row_bytes
    if minimum > memory_bytes:
        raise ValueError(
            f'RP memory budget for {geometry}: need at least {minimum} bytes, '
            f'configured {memory_bytes}; increase {memory_option}.'
        )
    spare = memory_bytes - fixed_bytes
    component_rows = max(units, (spare // 2) // COMPONENT_ROW_BYTES)
    max_component = component_rows // max(1, units)
    component_bytes = units * max_component * COMPONENT_ROW_BYTES
    effective_batch = min(batch_size, (spare - component_bytes) // row_bytes)
    return RefractoryBuildPlan(
        units,
        bins,
        disk_bytes,
        fixed_bytes,
        component_bytes,
        max_component,
        effective_batch,
        row_bytes,
    )
