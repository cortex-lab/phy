"""Sparse real-dimension regression for bounded RP summary construction."""

import numpy as np

from phy.apps.large_recording.live_summaries import LiveSummaryEngine
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.refractory_engine import RefractoryEngine
from phy.apps.large_recording.refractory_resources import (
    DEFAULT_RP_MAX_BYTES,
    DEFAULT_RP_MEMORY_BYTES,
    plan_refractory_summaries,
)
from phy.apps.large_recording.refractory_store import build_refractory_summaries
from phy.apps.large_recording.tests.test_source import make_sorting


def test_sparse_carl_dimensions_build_and_read_all_unit_counters(tmp_path):
    units = 962
    bins = 2052
    samples_per_bin = 1800 * 30000
    duration = bins * samples_per_bin
    params = make_sorting(tmp_path / 'source')

    # One globally ordered event per unit keeps the fixture sparse while the
    # scientific counter grids exercise the full unit-by-bin dimensions.
    np.save(params.parent / 'spike_times.npy', np.arange(units, dtype=np.int64) * 1000)
    np.save(params.parent / 'spike_clusters.npy', np.arange(units, dtype=np.int32))
    np.save(params.parent / 'spike_templates.npy', np.zeros(units, dtype=np.int32))
    np.save(params.parent / 'amplitudes.npy', np.ones(units, dtype=np.float32))
    for name in (
        'spike_positions.npy',
        'spike_depths.npy',
        'pc_features.npy',
        'pc_feature_ind.npy',
    ):
        (params.parent / name).unlink(missing_ok=True)
    for path in params.parent.glob('_phy_spikes_subset.*.npy'):
        path.unlink()
    params.write_text(
        params.read_text()
        .replace('sample_rate = 1.0', 'sample_rate = 30000.0')
        .replace('num_samples = 20', f'num_samples = {duration}')
    )

    bounded = prepare_large(params, tmp_path / 'bounded', max_spikes=units)
    progress = []
    output = build_refractory_summaries(
        bounded,
        batch_size=10_000_000,
        progress=lambda message, *args: progress.append(message),
    )

    with LiveSummaryEngine(bounded) as source:
        plan = plan_refractory_summaries(
            source.manifest,
            batch_size=10_000_000,
            native_grid_bytes=source.edges.nbytes + source.exposure.nbytes,
        )
    assert plan.units == units
    assert plan.bins == bins
    assert plan.batch_size < 10_000_000
    assert plan.memory_bytes <= DEFAULT_RP_MEMORY_BYTES
    assert plan.disk_bytes <= DEFAULT_RP_MAX_BYTES
    plan_message = next(message for message in progress if 'RP plan:' in message)
    assert f'batch {plan.batch_size}' in plan_message

    artifact_bytes = sum(path.stat().st_size for path in output.iterdir())
    assert artifact_bytes <= plan.disk_bytes <= DEFAULT_RP_MAX_BYTES

    params.parent.rename(tmp_path / 'offline-source')

    def no_source(*args, **kwargs):
        raise AssertionError('Prepared RP counters must work without source files.')

    with LiveSummaryEngine(bounded) as reader:
        reader._open = reader._unchanged = no_source
        engine = RefractoryEngine(reader)
        for unit in range(units):
            result = engine.resolve({unit: {'kind': 'source', 'unit': unit}})[unit]
            assert result['status'] == 'ready'
            expected_spikes = np.zeros(bins, dtype=np.int64)
            expected_spikes[0] = 1
            np.testing.assert_array_equal(result['spikes'], expected_spikes)
            np.testing.assert_array_equal(result['violations'], np.zeros(bins, dtype=np.int64))
            assert result['duplicates_removed'] == 0
