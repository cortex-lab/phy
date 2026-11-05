"""Independent discontinuous-clock preparation and replay oracle."""

import json
from contextlib import closing

import numpy as np
import pytest

from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import Source, hash_file
from phy.apps.large_recording.tests.test_source import make_sorting


def test_elapsed_producer_evidence_replay_and_exposure(tmp_path):
    params = make_sorting(tmp_path / 'source')
    contract = {'kind': 'elapsed_ticks', 'ticks_per_second': 1.0}
    intervals = [[0, 7], [10, 20]]
    params.write_text(
        params.read_text() + f'time_base = {contract!r}\nvalid_intervals = {intervals!r}\n'
    )
    source_path = params.parent
    samples = np.load(source_path / 'spike_times.npy')
    labels = np.load(source_path / 'spike_clusters.npy')
    waveform_ids = np.load(source_path / '_phy_spikes_subset.spikes.npy')
    np.save(source_path / '_phy_spikes_subset.sample_index.npy', samples[waveform_ids])
    np.save(source_path / '_phy_spikes_subset.window_bounds.npy', np.array(intervals, 'int64'))
    (source_path / 'timestamps.json').write_text('{"provenance": "producer-owned clock"}\n')
    (source_path / '_phy_spikes_subset.manifest.json').write_text(
        json.dumps(
            {
                'format': 'phy_spikes_subset',
                'format_version': 1,
                'recording': {
                    'time_base': contract,
                    'window_bounds_coordinates': 'elapsed_ticks',
                    'sample_index_coordinates': 'elapsed_ticks',
                },
            }
        )
    )
    original = {p.name: hash_file(p) for p in source_path.iterdir() if p.is_file()}
    bounded = prepare_large(params, tmp_path / 'bounded', bin_seconds=5)
    manifest = json.loads((bounded / '_phy_curation_manifest.json').read_text())
    assert manifest['time_base'] == contract
    assert manifest['selection']['resolved_window_policy'] == 'producer'
    np.testing.assert_array_equal(np.load(bounded / '_phy_spike_global_ids.npy'), np.arange(12))
    np.testing.assert_array_equal(
        np.load(bounded / '_phy_spikes_subset.sample_index.npy'), samples[waveform_ids]
    )
    # Independent exposure oracle, including the partial bin before the gap.
    np.testing.assert_array_equal(np.load(bounded / manifest['exposure_file']), [5, 2, 5, 5])
    counts = np.load(bounded / manifest['counts_file'])
    for row, unit in enumerate(manifest['unit_ids']):
        expected, _ = np.histogram(samples[labels == unit], bins=[0, 5, 10, 15, 20])
        np.testing.assert_array_equal(counts[row], expected)
    with closing(BoundedSession(bounded)) as session:
        session.record_merge([0, 1], 100)
        session.save()
    with closing(BoundedSession(bounded)) as reopened:
        np.testing.assert_array_equal(reopened.assignments, np.where(labels < 2, 100, labels))
    output = apply_curation(params, bounded, tmp_path / 'curated')
    np.testing.assert_array_equal(
        np.load(output / 'spike_clusters.npy'), np.where(labels < 2, 100, labels)
    )
    assert (output / 'timestamps.json').read_bytes() == (
        source_path / 'timestamps.json'
    ).read_bytes()
    with Source(params) as source, Source(output / 'params.py') as curated:
        assert source.identity == curated.identity
    again = prepare_large(output / 'params.py', tmp_path / 'again', bin_seconds=5)
    with closing(BoundedSession(again)) as session:
        assert session.manifest['time_base'] == contract
    assert original == {p.name: hash_file(p) for p in source_path.iterdir() if p.is_file()}
    # A metadata edit cannot change the coordinate interpretation of saved evidence.
    path = again / '_phy_curation_manifest.json'
    altered = json.loads(path.read_text())
    altered['time_base']['kind'] = 'acquisition_samples'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='time base'):
        BoundedSession(again)
