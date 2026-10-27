"""Live memberships and exact summaries compared with independent full replay."""

import json
from concurrent.futures import CancelledError

import numpy as np
import pytest

from phy.apps.large_recording.curation import project_assignments
from phy.apps.large_recording.live_summaries import LiveSummaryEngine, membership_expressions
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.tests.test_source import make_sorting


def split(parent, children, attribute='amplitudes.npy', threshold=4, nonfinite='keep'):
    return {
        'version': 1,
        'operation': 'threshold_split',
        'parent': parent,
        'children': children,
        'metadata': {},
        'predicate': {
            'attribute': attribute,
            'operator': '>=',
            'threshold': threshold,
            'nonfinite': nonfinite,
        },
    }


def merge(parents, target):
    return {
        'version': 1,
        'operation': 'merge',
        'parents': parents,
        'target': target,
        'metadata': {},
    }


def package(tmp_path, columns=False, positions=False):
    params = make_sorting(tmp_path / 'source', columns=columns)
    amplitude = np.load(params.parent / 'amplitudes.npy')
    amplitude[1], amplitude[8] = np.nan, np.inf
    np.save(params.parent / 'amplitudes.npy', amplitude[:, None] if columns else amplitude)
    if positions:
        (params.parent / 'spike_depths.npy').unlink()
        params.write_text(params.read_text() + 'depth_axis = 1\n')
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        max_spikes=8,
        window_seconds=3,
        stratum_seconds=20,
        bin_seconds=5,
        batch_size=2,
        valid_intervals=[[0, 7], [10, 20]],
    )
    return params.parent, bounded


def oracle(source, engine, events, unit):
    assignments = np.load(source / 'spike_clusters.npy').reshape(-1)
    attrs = {
        name: np.load(source / name).reshape(-1)
        for name in ('amplitudes.npy', 'spike_templates.npy')
    }
    assignments = project_assignments(assignments, events, attrs)
    times = np.load(source / 'spike_times.npy').reshape(-1)
    depth = (
        np.load(source / 'spike_depths.npy').reshape(-1)
        if (source / 'spike_depths.npy').exists()
        else np.load(source / 'spike_positions.npy')[:, 1]
    )
    result = engine._empty()
    for index in range(len(times)):
        if assignments[index] != unit:
            continue
        column = next(
            i
            for i, (a, b) in enumerate(zip(engine.edges[:-1], engine.edges[1:]))
            if a <= times[index] < b
        )
        result['counts'][column] += 1
        for field, values in [('amplitude', attrs['amplitudes.npy']), ('depth', depth)]:
            if np.isfinite(values[index]):
                result['trend_sums'][field][column] += values[index]
                result['trend_counts'][field][column] += 1
    return result


@pytest.mark.parametrize('columns,positions', [(False, False), (True, True)])
def test_exact_nested_split_merge_native_bins_and_offline_cache(tmp_path, columns, positions):
    source, bounded = package(tmp_path, columns, positions)
    before = {p.name: hash_file(p) for p in source.iterdir()}
    manifest_before = hash_file(bounded / '_phy_curation_manifest.json')
    events = [
        split(1, [100, 101]),
        merge([101, 2], 102),
        split(102, [103, 104], 'spike_templates.npy', 1),
        merge([100, 103], 105),
    ]
    expressions = membership_expressions([0, 1, 2, 99], events)
    with LiveSummaryEngine(bounded, batch_size=2) as engine:
        result = engine.resolve(expressions)
        np.testing.assert_array_equal(engine.exposure, [5, 2, 5, 5])
        for unit, summary in result.items():
            expected = oracle(source, engine, events, unit)
            for name, values in engine._flat(summary).items():
                np.testing.assert_allclose(values, engine._flat(expected)[name])
                assert not values.flags.writeable
        assert set(engine.arrays) == {
            'spike_times.npy',
            'spike_clusters.npy',
            'amplitudes.npy',
            'spike_templates.npy',
            'spike_positions.npy' if positions else 'spike_depths.npy',
        }
        engine.remember(expressions, result)
        assert len(engine.resident) * engine.summary_bytes <= engine.max_bytes
    assert {p.name: hash_file(p) for p in source.iterdir()} == before
    assert hash_file(bounded / '_phy_curation_manifest.json') == manifest_before
    source.rename(tmp_path / 'relocated')
    with LiveSummaryEngine(bounded) as engine:
        cached = engine.cached(expressions)
        assert set(cached) == set(expressions)
        assert not engine.arrays
        combined = {'kind': 'union', 'parts': [expressions[104], expressions[105]]}
        summary = engine.cached({200: combined})[200]
        np.testing.assert_array_equal(
            summary['counts'], result[104]['counts'] + result[105]['counts']
        )


def test_membership_identity_ignores_labels_and_transient_ids():
    a = membership_expressions([0, 1], [split(0, [2, 3])])
    b = membership_expressions(
        [0, 1],
        [
            split(0, [20, 30]),
            {'operation': 'label', 'ids': [30], 'field': 'group', 'value': 'good'},
        ],
    )
    assert a[2] == b[20] and a[3] == b[30]
    assert (
        membership_expressions([0, 1], [merge([0, 1], 2)])[2]
        == (membership_expressions([0, 1], [merge([1, 0], 20)])[20])
    )


def test_reject_only_parent_rows_and_keep_nonfinite_remainder(tmp_path):
    source, bounded = package(tmp_path)
    with LiveSummaryEngine(bounded, batch_size=1) as engine:
        events = [split(0, [100, 101], nonfinite='reject')]
        result = engine.resolve(membership_expressions([0, 1, 2, 99], events))
        assert sum(result[u]['counts'].sum() for u in (100, 101)) == 4
        with pytest.raises(ValueError, match='Nonfinite'):
            engine.resolve(
                membership_expressions([0, 1, 2, 99], [split(1, [100, 101], nonfinite='reject')])
            )
        assert not engine.arrays


@pytest.mark.parametrize('change', ['scalar', 'assignments', 'metadata', 'shape', 'missing'])
def test_source_changes_never_publish_completed_summary(tmp_path, change):
    source, bounded = package(tmp_path)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
    if change == 'metadata':
        (source / 'cluster_group.tsv').write_text('cluster_id\tgroup\n0\tnoise\n')
    else:
        name = 'spike_clusters.npy' if change == 'assignments' else 'amplitudes.npy'
        path = source / name
        array = np.load(path)
        if change == 'missing':
            path.unlink()
        else:
            array[0] = 99
            np.save(path, array[:-1] if change == 'shape' else array)
    with LiveSummaryEngine(bounded) as engine:
        with pytest.raises((ValueError, FileNotFoundError)):
            engine.resolve(expressions)
        assert not engine.arrays and not engine.handles
    assert not list((bounded / '.phy' / 'live-summary-cache').glob('*.npz'))


def test_cache_tamper_is_miss_and_relocation_supported(tmp_path):
    source, bounded = package(tmp_path)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
    with LiveSummaryEngine(bounded) as engine:
        result = engine.resolve(expressions)
        cache = engine.cache_dir / f'{engine.key(expressions[100])}.npz'
        with np.load(cache) as record:
            content = {name: record[name] for name in record.files}
        content['counts'] = content['counts'] + 5
        np.savez(cache, **content)
    relocated = tmp_path / 'moved'
    source.rename(relocated)
    with LiveSummaryEngine(bounded, source_dir=relocated) as engine:
        assert 100 not in engine.cached(expressions)
        actual = engine.resolve(expressions)
        np.testing.assert_array_equal(actual[100]['counts'], result[100]['counts'])


def test_cancellation_and_output_disk_budgets(tmp_path):
    source, bounded = package(tmp_path)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
    with LiveSummaryEngine(bounded, max_bytes=1) as engine:
        with pytest.raises(ValueError, match='budget'):
            engine.resolve(expressions)
        assert not engine.arrays
    with LiveSummaryEngine(bounded, cache_bytes=1) as engine:
        with pytest.raises(CancelledError):
            engine.resolve(expressions, cancel=lambda: True)
        assert not engine.arrays
        engine.resolve(expressions)
        assert not engine.cache_dir.exists()


def test_pinned_source_change_between_jobs_rejected(tmp_path):
    source, bounded = package(tmp_path)
    events = [split(1, [100, 101])]
    with LiveSummaryEngine(bounded) as engine:
        engine.resolve(membership_expressions([0, 1, 2, 99], events))
        path = source / 'amplitudes.npy'
        values = np.load(path)
        values[0] = 500
        np.save(path, values)
        events.append(split(0, [102, 103]))
        with pytest.raises(ValueError, match='changed'):
            engine.resolve(membership_expressions([0, 1, 2, 99], events))
        assert not engine.arrays


def test_reunited_children_restore_prepared_membership():
    expressions = membership_expressions([0, 1], [split(0, [2, 3]), merge([3, 2], 4)])
    assert expressions[4] == {'kind': 'source', 'unit': 0}


@pytest.mark.parametrize('phase', ['hash', 'scan'])
def test_mid_job_cancellation_releases_pinned_readers(tmp_path, phase):
    source, bounded = package(tmp_path)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
    with LiveSummaryEngine(bounded, batch_size=1) as engine:

        def cancel():
            if phase == 'hash':
                return bool(engine.handles)
            return bool(engine.arrays) and len(engine.arrays) >= 4

        with pytest.raises(CancelledError):
            engine.resolve(expressions, cancel)
        assert not engine.arrays and not engine.handles
        assert not list(engine.cache_dir.glob('*.npz'))


def test_same_job_source_replacement_never_publishes(tmp_path):
    source, bounded = package(tmp_path)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
    with LiveSummaryEngine(bounded, batch_size=1) as engine:
        replaced = []

        def mutate():
            if len(engine.arrays) >= 4 and not replaced:
                path = source / 'amplitudes.npy'
                old = path.with_suffix('.old')
                path.rename(old)
                np.save(path, np.load(old))
                replaced.append(True)
            return False

        with pytest.raises(ValueError, match='changed'):
            engine.resolve(expressions, mutate)
        assert not engine.arrays and not engine.handles
        assert not list(engine.cache_dir.glob('*.npz'))


@pytest.mark.parametrize('bin_seconds', [2, 6])
def test_zero_exposure_and_short_final_native_bins(tmp_path, bin_seconds):
    params = make_sorting(tmp_path / 'source')
    depth = np.load(params.parent / 'spike_depths.npy')
    depth[3], depth[7] = np.nan, -np.inf
    np.save(params.parent / 'spike_depths.npy', depth[:, None])
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        max_spikes=8,
        window_seconds=3,
        stratum_seconds=20,
        bin_seconds=bin_seconds,
        batch_size=2,
        valid_intervals=[[0, 7], [10, 20]],
    )
    events = [split(0, [100, 101])]
    with LiveSummaryEngine(bounded, batch_size=2) as engine:
        result = engine.resolve(membership_expressions([0, 1, 2, 99], events))
        for unit, summary in result.items():
            expected = oracle(params.parent, engine, events, unit)
            for name, values in engine._flat(summary).items():
                np.testing.assert_allclose(values, engine._flat(expected)[name])
        if bin_seconds == 2:
            assert engine.exposure[4] == 0
            assert all(summary['counts'][4] == 0 for summary in result.values())
        else:
            assert engine.edges[-1] - engine.edges[-2] == 2


def test_branching_memberships_match_full_projection(tmp_path):
    source, bounded = package(tmp_path)
    rng = np.random.default_rng(42)
    events = []
    active = [0, 1, 2, 99]
    next_id = 100
    for _ in range(18):
        if len(active) >= 3 and rng.random() < 0.4:
            parents = list(map(int, rng.choice(active, size=2, replace=False)))
            events.append(merge(parents, next_id))
            active = [unit for unit in active if unit not in parents] + [next_id]
            next_id += 1
        else:
            parent = int(rng.choice(active))
            events.append(
                split(
                    parent,
                    [next_id, next_id + 1],
                    'amplitudes.npy' if rng.random() < 0.5 else 'spike_templates.npy',
                    float(rng.integers(0, 10)),
                )
            )
            active = [unit for unit in active if unit != parent] + [next_id, next_id + 1]
            next_id += 2
    with LiveSummaryEngine(bounded, batch_size=2) as engine:
        for journal in [
            events,
            events[:9],
            events[:6] + [split(0, [1000, 1001])]
            if 0 in membership_expressions([0, 1, 2, 99], events[:6])
            else events[:4],
        ]:
            expressions = membership_expressions([0, 1, 2, 99], journal)
            results = engine.resolve(expressions)
            for unit, summary in results.items():
                expected = oracle(source, engine, journal, unit)
                for name, values in engine._flat(summary).items():
                    np.testing.assert_allclose(values, engine._flat(expected)[name])


def test_reject_split_reunion_still_checks_full_source_nonfinite(tmp_path):
    source, bounded = package(tmp_path)
    events = [split(1, [100, 101], nonfinite='reject'), merge([100, 101], 102)]
    with LiveSummaryEngine(bounded) as engine, pytest.raises(ValueError, match='Nonfinite'):
        engine.resolve(membership_expressions([0, 1, 2, 99], events))


@pytest.mark.parametrize('metadata', ['[]', 'null', '42', '{"key":null}'])
def test_valid_json_wrong_cache_metadata_is_disposable(tmp_path, metadata):
    source, bounded = package(tmp_path)
    expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
    with LiveSummaryEngine(bounded) as engine:
        expected = engine.resolve(expressions)
        cache = engine.cache_dir / f'{engine.key(expressions[100])}.npz'
        np.savez(cache, metadata=metadata)
        assert 100 not in engine.cached(expressions)
        actual = engine.resolve(expressions)
        np.testing.assert_array_equal(actual[100]['counts'], expected[100]['counts'])


def test_positions_validation_hashes_unselected_columns_too(tmp_path):
    source, bounded = package(tmp_path, positions=True)
    path = source / 'spike_positions.npy'
    positions = np.load(path)
    positions[0, 0] = 99
    np.save(path, positions)
    with LiveSummaryEngine(bounded) as engine:
        expressions = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])
        with pytest.raises(ValueError, match='hash mismatch'):
            engine.resolve(expressions)


def test_reader_reuses_validated_files_and_opens_new_predicates_only(tmp_path, monkeypatch):
    from pathlib import Path

    source, bounded = package(tmp_path)
    original = Path.open
    opened = []

    def record(path, *args, **kwargs):
        if path.parent == source and path.suffix == '.npy':
            opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', record)
    events = [split(1, [100, 101])]
    with LiveSummaryEngine(bounded) as engine:
        engine.resolve(membership_expressions([0, 1, 2, 99], events))
        first = list(opened)
        events.append(split(0, [102, 103], 'spike_templates.npy', 1))
        engine.resolve(membership_expressions([0, 1, 2, 99], events))
        assert opened[len(first) :] == ['spike_templates.npy']
        assert opened.count('spike_times.npy') == 1
        assert opened.count('spike_clusters.npy') == 1
        (source / 'cluster_group.tsv').write_text('cluster_id\tgroup\n0\tnoise\n')
        events.append(split(99, [104, 105]))
        with pytest.raises(ValueError, match='metadata changed'):
            engine.resolve(membership_expressions([0, 1, 2, 99], events))
        assert not engine.arrays


def test_disk_cache_growth_and_resident_eviction_are_bounded(tmp_path):
    source, bounded = package(tmp_path)
    with LiveSummaryEngine(bounded, cache_bytes=17000, max_bytes=500) as engine:
        events = [split(1, [100, 101])]
        first = membership_expressions([0, 1, 2, 99], events)
        first = {100: first[100], 101: first[101]}
        result = engine.resolve(first)
        engine.remember(first, result)
        events.append(split(0, [102, 103]))
        second = membership_expressions([0, 1, 2, 99], events)
        second = {102: second[102], 103: second[103]}
        more = engine.resolve(second)
        engine.remember(second, more)
        assert len(engine.resident) * engine.summary_bytes <= engine.max_bytes
        assert (
            sum(path.stat().st_size for path in engine.cache_dir.glob('*.npz'))
            <= engine.cache_bytes
        )


def test_cache_header_claiming_huge_array_is_rejected_before_numpy_load(tmp_path, monkeypatch):
    import io
    import zipfile

    source, bounded = package(tmp_path)
    expression = membership_expressions([0, 1, 2, 99], [split(1, [100, 101])])[100]
    with LiveSummaryEngine(bounded) as engine:
        engine.resolve({100: expression})
        cache = engine.cache_dir / f'{engine.key(expression)}.npz'
        with zipfile.ZipFile(cache) as archive:
            entries = {info.filename: archive.read(info) for info in archive.infolist()}
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header, {'descr': '<i8', 'fortran_order': False, 'shape': (2**40,)}
        )
        entries['counts.npy'] = header.getvalue()
        with zipfile.ZipFile(cache, 'w') as archive:
            for name, payload in entries.items():
                archive.writestr(name, payload)
        original = np.load

        def guarded(path, *args, **kwargs):
            if hasattr(path, 'read') or str(path) == str(cache):
                pytest.fail('Malformed cache reached numpy allocation.')
            return original(path, *args, **kwargs)

        monkeypatch.setattr(np, 'load', guarded)
        assert engine.cached({100: expression}) == {}
