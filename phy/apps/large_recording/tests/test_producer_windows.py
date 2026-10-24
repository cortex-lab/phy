"""Producer plans retain complete windows and preserve sparse waveform row alignment."""

import json

import numpy as np
import pytest

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.tests.test_source import make_sorting

STEM = '_phy_spikes_subset.'


def producer_sorting(path):
    params = make_sorting(path)
    bounds = np.array([[0, 3], [5, 7], [10, 12], [14, 15], [18, 20]])
    np.save(path / (STEM + 'window_bounds.npy'), bounds)
    np.save(path / (STEM + 'window_types.npy'), np.zeros(5, dtype=np.uint8))
    np.save(path / (STEM + 'window_strata.npy'), np.arange(5, dtype=np.int32))
    return params


def read_manifest(path):
    return json.loads((path / '_phy_curation_manifest.json').read_text())


def test_auto_prefers_producer_and_exact_waveform_remapping(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    output = prepare_large(params, tmp_path / 'bounded', window_policy='producer', batch_size=2)
    manifest = read_manifest(output)
    assert manifest['selection']['resolved_window_policy'] == 'producer'
    assert manifest['waveform_coverage'] == {
        'source_rows': 5,
        'retained_rows': 5,
        'dropped_rows': 0,
    }
    np.testing.assert_array_equal(
        np.load(output / manifest['windows_file']),
        np.load(params.parent / (STEM + 'window_bounds.npy')),
    )
    ids = np.load(output / manifest['global_spike_ids_file'])
    upstream_ids = np.load(params.parent / (STEM + 'spikes.npy'))
    np.testing.assert_array_equal(ids[np.load(output / (STEM + 'spikes.npy'))], upstream_ids)
    for suffix in ('waveforms.npy', 'channels.npy'):
        np.testing.assert_array_equal(
            np.load(output / (STEM + suffix)), np.load(params.parent / (STEM + suffix))
        )
    assert manifest['producer_window_plan']['sha256']
    assert manifest['producer_window_estimate']['spikes'] == 12
    auto = prepare_large(params, tmp_path / 'auto')
    assert read_manifest(auto)['selection']['resolved_window_policy'] == 'producer'


def test_budget_distributes_complete_windows_across_recording(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    output = prepare_large(params, tmp_path / 'bounded', window_policy='producer', max_spikes=7)
    manifest = read_manifest(output)
    # Dyadic traversal takes the first and last windows before exhausting budget.
    assert manifest['bounded_num_spikes'] == 7
    assert [(r['start'], r['end']) for r in manifest['accepted_window_candidates']] == [
        (0, 3),
        (18, 20),
    ]
    assert manifest['omitted_windows']
    assert all(r['reason'] == 'spike_budget' for r in manifest['omitted_windows'])
    assert manifest['waveform_coverage']['dropped_rows'] > 0


def test_generated_override_and_missing_producer(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    output = prepare_large(
        params,
        tmp_path / 'bounded',
        window_policy='generated',
        window_seconds=20,
        stratum_seconds=20,
    )
    assert read_manifest(output)['selection']['resolved_window_policy'] == 'generated'
    params = make_sorting(tmp_path / 'legacy')
    with pytest.raises(ValueError, match='requires nonempty'):
        prepare_large(params, tmp_path / 'missing', window_policy='producer')
    with pytest.raises(ValueError, match='window_policy'):
        prepare_large(params, tmp_path / 'bad', window_policy='invalid')


@pytest.mark.parametrize(
    'name,value,match',
    [
        ('window_bounds.npy', np.array([0, 3]), 'integer'),
        ('window_bounds.npy', np.array([[0.0, 3.0]]), 'integer'),
        ('window_bounds.npy', np.array([[3, 2]]), 'bounds'),
        ('window_bounds.npy', np.array([[0, 21]]), 'bounds'),
        ('window_bounds.npy', np.array([[5, 6], [0, 3]]), 'sorted'),
        ('window_types.npy', np.zeros(4, dtype=np.uint8), 'aligned'),
        ('window_types.npy', np.full(5, 2, dtype=np.uint8), 'codes'),
        ('window_strata.npy', np.full(5, -2), 'strata'),
    ],
)
def test_invalid_sidecars_rejected_even_for_generated(tmp_path, name, value, match):
    params = producer_sorting(tmp_path / 'source')
    np.save(params.parent / (STEM + name), value)
    if name == 'window_bounds.npy':
        for suffix in ('window_types.npy', 'window_strata.npy'):
            (params.parent / (STEM + suffix)).unlink()
    with pytest.raises(ValueError, match=match):
        prepare_large(params, tmp_path / 'bounded', window_policy='generated')


@pytest.mark.parametrize('intervals', [[[0, 10], [10, 20]], [[0, 9], [10, 20]]])
def test_producer_window_cannot_cross_block_or_gap(tmp_path, intervals):
    params = producer_sorting(tmp_path / 'source')
    with params.open('a') as stream:
        stream.write(f'valid_intervals = {intervals!r}\n')
    np.save(params.parent / (STEM + 'window_bounds.npy'), np.array([[0, 11]]))
    for suffix in ('window_types.npy', 'window_strata.npy'):
        (params.parent / (STEM + suffix)).unlink()
    with pytest.raises(ValueError, match='block|gap'):
        prepare_large(params, tmp_path / 'bounded')


def test_overlaps_union_without_duplicate_spikes(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    np.save(params.parent / (STEM + 'window_bounds.npy'), np.array([[0, 12], [10, 20]]))
    for suffix in ('window_types.npy', 'window_strata.npy'):
        (params.parent / (STEM + suffix)).unlink()
    output = prepare_large(params, tmp_path / 'bounded', window_policy='producer')
    manifest = read_manifest(output)
    assert manifest['bounded_num_spikes'] == 12
    np.testing.assert_array_equal(np.load(output / manifest['windows_file']), [[0, 20]])


def test_waveforms_outside_producer_windows_rejected(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    np.save(params.parent / (STEM + 'window_bounds.npy'), np.array([[0, 3]]))
    for suffix in ('window_types.npy', 'window_strata.npy'):
        (params.parent / (STEM + suffix)).unlink()
    with pytest.raises(ValueError, match='outside'):
        prepare_large(params, tmp_path / 'bounded', window_policy='producer')


def test_wrong_window_coordinates_rejected_for_generated(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    (params.parent / (STEM + 'manifest.json')).write_text(
        json.dumps(
            {
                'format': 'phy_spikes_subset',
                'format_version': 1,
                'recording': {'window_bounds_coordinates': 'elapsed_ticks'},
            }
        )
    )
    with pytest.raises(ValueError, match='coordinates|time_base'):
        prepare_large(params, tmp_path / 'bounded', window_policy='generated')


def test_producer_policy_requires_waveform_triplet(tmp_path):
    params = producer_sorting(tmp_path / 'source')
    for suffix in ('spikes.npy', 'channels.npy', 'waveforms.npy'):
        (params.parent / (STEM + suffix)).unlink()
    with pytest.raises(ValueError, match='requires nonempty'):
        prepare_large(params, tmp_path / 'bounded', window_policy='producer')
