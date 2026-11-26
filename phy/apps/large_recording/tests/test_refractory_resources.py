"""Allocation planning at real long-recording dimensions and budget boundaries."""

import pytest

from phy.apps.large_recording.refractory_resources import (
    COMPONENT_ROW_BYTES,
    plan_refractory_summaries,
)


def geometry(units=962, bins=2052):
    return {
        'sample_rate': 30000,
        'num_samples': bins * 1800 * 30000,
        'unit_ids': list(range(units)),
        'valid_intervals': [[0, bins * 1800 * 30000]],
    }


def test_carl_dimensions_fit_defaults_and_batch_is_bounded():
    plan = plan_refractory_summaries(geometry(), batch_size=10000000)
    assert plan.disk_bytes < 67108864
    assert plan.memory_bytes <= 67108864
    assert plan.batch_size < 10000000
    assert plan.max_component >= 1
    assert plan.disk_bytes >= 16 * 962 * 2052


def test_exact_disk_and_minimum_memory_boundaries():
    manifest = geometry(3, 2)
    plan = plan_refractory_summaries(manifest)
    minimum = plan.fixed_bytes + 3 * COMPONENT_ROW_BYTES + plan.row_bytes
    with pytest.raises(ValueError, match=r'disk budget.*--max-bytes'):
        plan_refractory_summaries(manifest, max_bytes=plan.disk_bytes - 1)
    exact = plan_refractory_summaries(manifest, max_bytes=plan.disk_bytes, memory_bytes=minimum)
    assert exact.batch_size == exact.max_component == 1
    assert exact.memory_bytes == minimum
    with pytest.raises(ValueError, match=r'memory budget.*--rp-summary-memory-bytes'):
        plan_refractory_summaries(
            manifest, memory_bytes=minimum - 1, memory_option='--rp-summary-memory-bytes'
        )


def test_dtype_and_native_grid_scratch_are_accounted():
    manifest = geometry(3, 2)
    baseline = plan_refractory_summaries(manifest, batch_size=10000000)
    manifest['source_array_inventory'] = {'amplitudes.npy': {'dtype': 'float128'}}
    wide = plan_refractory_summaries(manifest, native_grid_bytes=4096, batch_size=10000000)
    assert wide.fixed_bytes == baseline.fixed_bytes + 4096
    assert wide.row_bytes > baseline.row_bytes
    assert wide.batch_size < baseline.batch_size


def test_fragmented_exposure_metadata_is_budgeted_with_few_bins():
    manifest = geometry(1, 1)
    baseline = plan_refractory_summaries(manifest)
    manifest['valid_intervals'] = [[i * 2, i * 2 + 1] for i in range(100000)]
    fragmented = plan_refractory_summaries(manifest)
    assert fragmented.fixed_bytes >= baseline.fixed_bytes + 128 * 99999
    with pytest.raises(ValueError, match='memory budget'):
        plan_refractory_summaries(
            manifest, memory_bytes=baseline.fixed_bytes + COMPONENT_ROW_BYTES + baseline.row_bytes
        )


@pytest.mark.parametrize('key,value', [('sample_rate', 0), ('num_samples', 0)])
def test_invalid_scientific_geometry(key, value):
    manifest = geometry()
    manifest[key] = value
    with pytest.raises(ValueError):
        plan_refractory_summaries(manifest)
