"""Full counts and acquisition exposure compared with an independent tiny oracle."""

import numpy as np
import pytest

from phy.apps.large_recording.source import Source
from phy.apps.large_recording.summaries import build_summaries
from phy.apps.large_recording.tests.test_source import make_sorting


@pytest.mark.parametrize('workers', [1, 3])
def test_exact_counts_exposure_and_additive_measurements(tmp_path, workers):
    params = make_sorting(tmp_path / 'full', columns=True)
    amplitude = np.load(params.parent / 'amplitudes.npy')
    amplitude[1] = np.nan
    np.save(params.parent / 'amplitudes.npy', amplitude[:, None])
    source = Source(params, valid_intervals=[[0, 7], [10, 20]], batch_size=2)
    files = build_summaries(source, tmp_path / 'summary', bin_seconds=5, workers=workers)
    counts = np.load(tmp_path / 'summary' / files['counts_file'])
    expected = np.zeros((4, 4), dtype=np.int64)
    expected_sum = np.zeros((4, 4))
    expected_n = np.zeros((4, 4), dtype=np.int64)
    units = np.load(params.parent / 'spike_clusters.npy').reshape(-1)
    times = np.load(params.parent / 'spike_times.npy').reshape(-1)
    for unit, time, value in zip(units, times, amplitude, strict=True):
        row = list(source.unit_ids).index(unit)
        column = int(time // 5)
        expected[row, column] += 1
        if np.isfinite(value):
            expected_sum[row, column] += value
            expected_n[row, column] += 1
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(counts.sum(axis=1), source.counts)
    np.testing.assert_array_equal(
        np.load(tmp_path / 'summary' / files['edges_file']), [0, 5, 10, 15, 20]
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / 'summary' / files['exposure_file']), [5, 2, 5, 5]
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / 'summary' / files['amplitude_sum_file']), expected_sum
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / 'summary' / files['amplitude_count_file']), expected_n
    )
    assert files['measurement_units']['amplitude'] == 'source amplitude units'
    assert 'depth_sum_file' in files


def test_summary_budget_checked_before_output_allocation(tmp_path):
    source = Source(make_sorting(tmp_path / 'full'))
    with pytest.raises(ValueError, match='Summary budget'):
        build_summaries(source, tmp_path / 'summary', bin_seconds=1, max_bytes=50)
    assert not (tmp_path / 'summary').exists()


def test_position_coordinates_require_declared_depth_axis(tmp_path):
    from phy.apps.large_recording.summaries import scalar_attributes

    params = make_sorting(tmp_path / 'full')
    (params.parent / 'spike_depths.npy').unlink()
    with Source(params) as source:
        assert 'depth' not in scalar_attributes(source)
    params.write_text(params.read_text() + 'depth_axis = 1\n')
    with Source(params) as source:
        np.testing.assert_array_equal(
            scalar_attributes(source)['depth'], source.arrays['spike_positions.npy'][:, 1]
        )


def test_sparse_ids_gaps_and_partial_last_bin(tmp_path):
    params = make_sorting(tmp_path / 'full')
    units = np.load(params.parent / 'spike_clusters.npy')
    units[units == 2] = 2147483647
    np.save(params.parent / 'spike_clusters.npy', units)
    with Source(params, batch_size=2) as source:
        files = build_summaries(source, tmp_path / 'summary', bin_seconds=3)
        counts = np.load(tmp_path / 'summary' / files['counts_file'])
        np.testing.assert_array_equal(counts.sum(axis=1), source.counts)
        np.testing.assert_array_equal(
            np.load(tmp_path / 'summary' / files['edges_file']), [0, 3, 6, 9, 12, 15, 18, 20]
        )
        np.testing.assert_array_equal(
            np.load(tmp_path / 'summary' / files['exposure_file']), [3, 3, 3, 3, 3, 3, 2]
        )


def test_summary_tight_budget_and_parallel_outputs_agree(tmp_path):
    from phy.apps.large_recording.summaries import summary_size

    with Source(make_sorting(tmp_path / 'full'), batch_size=3) as source:
        size, _, _ = summary_size(source, bin_seconds=5)
        outputs = []
        for workers in (1, 3):
            directory = tmp_path / str(workers)
            files = build_summaries(
                source, directory, bin_seconds=5, max_bytes=size + 5120, workers=workers
            )
            outputs.append(
                {
                    key: np.load(directory / name)
                    for key, name in files.items()
                    if key.endswith('_file')
                }
            )
        for name in outputs[0]:
            np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
