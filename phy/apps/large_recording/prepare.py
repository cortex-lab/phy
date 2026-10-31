"""Publish a self-contained bounded evidence package from a full sorting."""

import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np

from .metadata import POLICY, SCHEMA_FILE, write_metadata
from .source import (
    PER_SPIKE,
    STATIC,
    WAVEFORMS,
    Source,
    close_array,
    hash_file,
    hash_json,
    publish_directory,
)
from .summaries import build_summaries, scalar_attributes, summary_size


def _row_bounds(times, windows):
    return [
        (
            int(np.searchsorted(times, start, side='left')),
            int(np.searchsorted(times, end, side='left')),
        )
        for start, end in windows
    ]


def _union(windows, block_starts=()):
    result = []
    for start, end in sorted(windows, key=lambda row: (int(row[0]), int(row[1]))):
        if result and (
            start < result[-1][1] or (start == result[-1][1] and start not in block_starts)
        ):
            result[-1][1] = max(result[-1][1], int(end))
        else:
            result.append([int(start), int(end)])
    return result


def _selected_waveform_count(source, windows):
    ids = source.arrays.get('_phy_spikes_subset.spikes.npy')
    if ids is None:
        return 0
    return sum(
        int(np.searchsorted(ids, stop) - np.searchsorted(ids, start))
        for start, stop in _row_bounds(source.arrays['spike_times.npy'].reshape(-1), windows)
    )


def _producer_windows(source):
    """Validate optional producer plans without loading waveform tensors."""
    stem = '_phy_spikes_subset.'
    bounds = source.arrays.get(stem + 'window_bounds.npy')
    types = source.arrays.get(stem + 'window_types.npy')
    strata = source.arrays.get(stem + 'window_strata.npy')
    if bounds is None:
        if types is not None or strata is not None:
            raise ValueError('Producer window types/strata require window_bounds.')
        return None, None
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.dtype.kind not in 'iu':
        raise ValueError('Producer window_bounds must be an integer (W, 2) array.')
    for label, values in (('types', types), ('strata', strata)):
        if values is not None and (
            values.ndim != 1 or len(values) != len(bounds) or values.dtype.kind not in 'iu'
        ):
            raise ValueError(f'Producer window_{label} must be an aligned integer vector.')
    manifest_path = source.path / (stem + 'manifest.json')
    info = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    coordinates = info.get('recording', {}).get('window_bounds_coordinates')
    clock = getattr(source, 'time_base', None)
    kind = clock['kind'] if clock else 'acquisition_samples'
    if coordinates is not None and coordinates != kind:
        raise ValueError('Producer window_bounds_coordinates disagrees with the source time_base.')
    if kind == 'elapsed_ticks' and coordinates is None:
        raise ValueError('Elapsed-tick producer windows require window_bounds_coordinates.')
    previous = -1
    for index, (first, last) in enumerate(bounds):
        first, last = int(first), int(last)
        block = int(np.searchsorted(source.valid_intervals[:, 0], first, side='right') - 1)
        if first < previous or first < 0 or last <= first or last > source.num_samples:
            raise ValueError(
                'Producer window bounds must be sorted, positive, and inside recording.'
            )
        if block < 0 or last > source.valid_intervals[block, 1]:
            raise ValueError('Producer window crosses an acquisition block or gap.')
        previous = first
        code = int(types[index]) if types is not None else 0
        stratum = int(strata[index]) if strata is not None else (-1 if code == 1 else 0)
        if code not in (0, 1):
            raise ValueError('Producer window_types codes must be 0 (survey) or 1 (rescue).')
        if stratum < -1 or (code == 1 and stratum != -1) or (code == 0 and stratum < 0):
            raise ValueError(
                'Producer window_strata must be nonnegative for survey and -1 for rescue.'
            )
    if source.arrays.keys() >= WAVEFORMS:
        union = _union(bounds, set(source.valid_intervals[:, 0]))
        if _selected_waveform_count(source, union) != len(source.arrays[stem + 'spikes.npy']):
            raise ValueError('Stored waveform rows fall outside the producer window plan.')
    provenance = {
        'sha256': hash_json(
            {
                'files': {
                    name: hash_file(source.path / name)
                    for name in (
                        stem + 'window_bounds.npy',
                        stem + 'window_types.npy',
                        stem + 'window_strata.npy',
                    )
                    if name in source.arrays
                },
                'coordinates': coordinates,
            }
        ),
        'window_bounds_coordinates': coordinates,
        'num_windows': len(bounds),
    }
    return (bounds, types, strata), provenance


def _distributed_indices(count):
    """Visit sorted windows in a reproducible dyadic spread, using constant memory."""
    if not count:
        return
    bits = (count - 1).bit_length()
    for value in range(1 << bits):
        index, remainder = 0, value
        for _ in range(bits):
            index = (index << 1) | (remainder & 1)
            remainder >>= 1
        if index < count:
            yield index


def prepare_large(
    params_path,
    output_dir,
    duration_seconds=None,
    valid_intervals=None,
    max_spikes=2000000,
    max_bytes=536870912,
    window_seconds=30,
    stratum_seconds=3600,
    seed=0,
    batch_size=100000,
    bin_seconds=3600,
    window_policy='auto',
    summary_store=None,
    summary_store_max_bytes=68719476736,
):
    """Choose complete shared windows, write bounded arrays, and publish atomically.

    Survey windows use a seeded offset within each acquisition stratum. A second
    pass tries one first-spike rescue window for every unit without evidence.
    Candidate windows that exceed either budget are explicitly omitted; spikes
    within accepted windows are never independently thinned.
    """
    with Source(params_path, duration_seconds, valid_intervals, batch_size) as source:
        output = _prepare_source(
            source,
            output_dir,
            max_spikes,
            max_bytes,
            window_seconds,
            stratum_seconds,
            seed,
            batch_size,
            bin_seconds,
            window_policy,
        )
    if summary_store is not None:
        from .summary_store import build_summary_store

        try:
            from .range_build import preflight_summary_store

            preflight_summary_store(output, summary_store_max_bytes)
            _, reference = build_summary_store(
                output,
                summary_store,
                source_dir=source.path,
                max_bytes=summary_store_max_bytes,
                return_reference=True,
            )
            from .range_build import build_amplitude_index
            from .range_intervals import RangeUnavailableError

            # Unsupported scalar dtypes keep the verified row snapshot.
            with suppress(RangeUnavailableError):
                build_amplitude_index(
                    output,
                    Path(summary_store) / 'amplitude-index',
                    max_bytes=summary_store_max_bytes,
                    expected_reference=reference,
                )
        except (ValueError, OSError) as error:
            raise ValueError(
                f'Bounded package is ready at {output}; optional summary store failed: '
                f'{error}. Use build-summary-store to retry without preparing again.'
            ) from error
    return output


def _prepare_source(
    source,
    output_dir,
    max_spikes,
    max_bytes,
    window_seconds,
    stratum_seconds,
    seed,
    batch_size,
    bin_seconds,
    window_policy,
):
    if window_policy not in ('auto', 'producer', 'generated'):
        raise ValueError('window_policy must be auto, producer, or generated.')
    producer, producer_provenance = _producer_windows(source)
    available = (
        producer is not None and bool(len(producer[0])) and source.arrays.keys() >= WAVEFORMS
    )
    if window_policy == 'producer' and not available:
        raise ValueError(
            'Producer policy requires nonempty window bounds and stored waveform data.'
        )
    resolved_policy = 'producer' if available and window_policy != 'generated' else 'generated'
    output_dir = Path(output_dir).resolve()
    missing_static = {
        'templates.npy',
        'channel_map.npy',
        'channel_positions.npy',
    } - source.arrays.keys()
    if missing_static:
        raise ValueError(f'Bounded GUI preparation requires: {", ".join(sorted(missing_static))}.')
    if output_dir.exists():
        raise ValueError(f'Output already exists: {output_dir}.')
    if output_dir == source.path or source.path in output_dir.parents:
        raise ValueError('Bounded output must be outside the source dataset directory.')
    if int(max_spikes) != max_spikes or max_spikes <= 0 or max_bytes <= 0:
        raise ValueError('max_spikes and max_bytes must be positive budgets.')
    widths = []
    for label, seconds in (
        ('window_seconds', window_seconds),
        ('stratum_seconds', stratum_seconds),
    ):
        width = float(seconds) * source.sample_rate
        if (
            not np.isfinite(width)
            or width < 1
            or not np.isclose(width, round(width), rtol=0, atol=1e-6)
        ):
            raise ValueError(f'{label} must specify a positive integer sample width.')
        widths.append(int(round(width)))
    width, stratum = widths
    times = source.arrays['spike_times.npy'].reshape(-1)
    assignments = source.arrays['spike_clusters.npy'].reshape(-1)
    summary_bytes = summary_size(source, bin_seconds)[0]
    producer_documents = [
        source.path / name
        for name in ('curation.json', 'curation_review.json')
        if (source.path / name).is_file()
    ]
    producer_hashes = {path.name: hash_file(path) for path in producer_documents}
    fixed_bytes = (
        sum(
            path.stat().st_size * (2 if path.name == 'curation_review.json' else 1)
            for path in producer_documents
        )
        + summary_bytes
        + sum((source.path / name).stat().st_size for name in source.arrays.keys() & STATIC)
    )
    # Reserve headers, manifest, params, metadata, and window records explicitly.
    metadata_bytes = sum(
        len(str(value).encode('utf-8')) + 64
        for values in source.metadata.values()
        for value in values.values()
    )
    overhead = 32768 + metadata_bytes + 1024 * len(source.arrays)
    fixed_bytes += overhead
    # Candidate records remain present even when rejected. Reserve them before
    # selection so an omission cannot silently exceed the byte budget.
    if resolved_policy == 'producer':
        fixed_bytes += len(producer[0]) * 512
    row_bytes = 16  # int64 global row mapping and baseline assignment
    for name in source.arrays.keys() & PER_SPIKE:
        array = source.arrays[name]
        row_bytes += array.dtype.itemsize * int(np.prod(array.shape[1:], dtype=np.int64))
    waveform_sidecars = source.arrays.keys() & {'_phy_spikes_subset.sample_index.npy'}
    waveform_bytes = 0
    if source.arrays.keys() >= WAVEFORMS:
        for name in WAVEFORMS | waveform_sidecars:
            array = source.arrays[name]
            waveform_bytes += array.dtype.itemsize * int(np.prod(array.shape[1:], dtype=np.int64))
    if fixed_bytes > max_bytes:
        raise ValueError(
            f'Output budget exceeded before selecting evidence: estimated '
            f'{fixed_bytes} bytes; increase max_bytes or bin_seconds.'
        )
    windows, plan, omitted = [], [], []
    rng = np.random.default_rng(seed)

    def accept(start, end, kind, segment, stratum_id):
        candidate = _union([*windows, [start, end]], set(source.valid_intervals[:, 0]))
        n = sum(stop - first for first, stop in _row_bounds(times, candidate))
        n_waveforms = _selected_waveform_count(source, candidate)
        size = (
            fixed_bytes
            + n * row_bytes
            + n_waveforms * waveform_bytes
            + (len(candidate) + len(plan) + len(omitted) + 1) * 256
        )
        record = {
            'start': int(start),
            'end': int(end),
            'kind': kind,
            'segment': 0,
            'block': int(segment),
            'stratum': int(stratum_id),
        }
        if (len(plan) + len(omitted) + 1) * 256 + fixed_bytes > max_bytes:
            raise ValueError('Window plan metadata exceeds max_bytes; increase max_bytes.')
        if n > max_spikes or size > max_bytes:
            omitted.append(
                {**record, 'reason': 'spike_budget' if n > max_spikes else 'byte_budget'}
            )
            return False
        windows[:] = candidate
        plan.append(record)
        return True

    producer_estimate = None
    if resolved_policy == 'producer':
        bounds, types, strata = producer
        # Counting intersections uses sparse ID binary searches, never waveform data.
        union = _union(bounds, set(source.valid_intervals[:, 0]))
        all_spikes = sum(stop - first for first, stop in _row_bounds(times, union))
        all_waveforms = _selected_waveform_count(source, union)
        producer_estimate = {
            'spikes': all_spikes,
            'waveform_rows': all_waveforms,
            'bytes': fixed_bytes
            + all_spikes * row_bytes
            + all_waveforms * waveform_bytes
            + (len(bounds) + len(union)) * 256,
        }
        del union
        for index in _distributed_indices(len(bounds)):
            first, last = map(int, bounds[index])
            block = int(np.searchsorted(source.valid_intervals[:, 0], first, side='right') - 1)
            code = int(types[index]) if types is not None else 0
            kind = 'survey' if code == 0 else 'rescue'
            stratum_id = int(strata[index]) if strata is not None else (-1 if code else index)
            accept(first, last, kind, block, stratum_id)
    else:
        # Candidate enumeration is scalar and bounded by output-window metadata,
        # rather than constructing recording-length arrays.
        for segment, (first, last) in enumerate(source.valid_intervals):
            for start in range(int(first), int(last), stratum):
                end = min(start + stratum, int(last))
                length = min(width, end - start)
                offset = int(rng.integers(0, end - start - length + 1))
                accept(
                    start + offset, start + offset + length, 'survey', segment, start // stratum
                )
                if (len(plan) + len(omitted)) * 256 + fixed_bytes > max_bytes:
                    raise ValueError(
                        'Window plan metadata exceeds max_bytes; increase stratum_seconds.'
                    )
    evidence = set()
    for first, stop in _row_bounds(times, windows):
        for start in range(first, stop, batch_size):
            evidence.update(np.unique(assignments[start : min(start + batch_size, stop)]).tolist())
    missing = set(source.unit_ids.tolist()) - evidence
    for batch in source.iter_batches():
        if window_policy == 'producer':
            break
        if not missing:
            break
        units, positions = np.unique(batch['spike_clusters.npy'], return_index=True)
        for unit, position in zip(units.tolist(), positions.tolist(), strict=True):
            if unit not in missing:
                continue
            sample = int(batch['spike_times.npy'][position])
            segment = int(np.searchsorted(source.valid_intervals[:, 0], sample, side='right') - 1)
            first, last = source.valid_intervals[segment]
            start = max(int(first), min(sample - width // 2, int(last) - width))
            end = min(start + width, int(last))
            accept(start, end, 'rescue', segment, -1)
            # One deterministic rescue attempt per unit, including rejected attempts.
            missing.remove(unit)
    row_bounds = _row_bounds(times, windows)
    n_bounded = sum(stop - first for first, stop in row_bounds)
    if not n_bounded:
        raise ValueError(
            'No bounded evidence fits the budgets; increase max_spikes/max_bytes '
            'or shorten window_seconds (tied spikes remain together).'
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = (
        fixed_bytes
        + n_bounded * row_bytes
        + _selected_waveform_count(source, windows) * waveform_bytes
        + (len(windows) + len(plan) + len(omitted)) * 256
    )
    free = shutil.disk_usage(output_dir.parent).free
    if free < estimated_bytes:
        raise OSError(
            f'Insufficient disk space for bounded preparation: estimated {estimated_bytes:,} bytes, {free:,} available.'
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=f'.{output_dir.name}.prepare-', dir=output_dir.parent)
    )
    global_ids, base, destinations = None, None, {}
    try:
        global_ids = np.lib.format.open_memmap(
            temporary / '_phy_spike_global_ids.npy', mode='w+', dtype=np.int64, shape=(n_bounded,)
        )
        for name in source.arrays.keys() & PER_SPIKE:
            array = source.arrays[name]
            destinations[name] = np.lib.format.open_memmap(
                temporary / name, mode='w+', dtype=array.dtype, shape=(n_bounded, *array.shape[1:])
            )
        base = np.lib.format.open_memmap(
            temporary / '_phy_spike_clusters_base.npy',
            mode='w+',
            dtype=np.int64,
            shape=(n_bounded,),
        )
        cursor = 0
        bounded_counts = {int(unit): 0 for unit in source.unit_ids}
        for first, stop in row_bounds:
            for start in range(first, stop, batch_size):
                end = min(start + batch_size, stop)
                length = end - start
                global_ids[cursor : cursor + length] = np.arange(start, end, dtype=np.int64)
                base[cursor : cursor + length] = assignments[start:end]
                for name, target in destinations.items():
                    target[cursor : cursor + length] = source.arrays[name][start:end]
                ids, counts = np.unique(assignments[start:end], return_counts=True)
                for unit, count in zip(ids.tolist(), counts.tolist(), strict=True):
                    bounded_counts[unit] += count
                cursor += length
        for array in (global_ids, base, *destinations.values()):
            array.flush()
        for name in source.arrays.keys() & STATIC:
            shutil.copyfile(source.path / name, temporary / name)
        if source.arrays.keys() >= WAVEFORMS:
            _copy_waveforms(source, temporary, global_ids, row_bounds)
        waveform_counts = {int(unit): 0 for unit in source.unit_ids}
        waveform_ids_path = temporary / '_phy_spikes_subset.spikes.npy'
        if waveform_ids_path.exists():
            waveform_ids = np.load(waveform_ids_path, mmap_mode='r')
            for start in range(0, len(waveform_ids), batch_size):
                ids, counts = np.unique(
                    base[waveform_ids[start : start + batch_size]], return_counts=True
                )
                for unit, count in zip(ids, counts):
                    waveform_counts[int(unit)] += int(count)
            waveform_ids._mmap.close()
        np.save(temporary / '_phy_curation_windows.npy', np.asarray(windows, dtype=np.int64))
        metadata = source.metadata
        write_metadata(temporary, metadata, source.metadata_schema)
        # Standard loader stays self-contained; no original raw path is referenced.
        params = {
            key: source.params[key]
            for key in (
                'sample_rate',
                'n_channels_dat',
                'dtype',
                'offset',
                'hp_filtered',
                'acquisition_id',
                'depth_axis',
                'amplitude_units',
                'depth_units',
                'position_units',
                'waveform_units',
            )
            if key in source.params
        }
        params.update(
            dat_path=None,
            sample_rate=source.sample_rate,
            num_samples=source.num_samples,
            valid_intervals=source.valid_intervals.tolist(),
        )
        if getattr(source, 'time_base', None) is not None:
            params['time_base'] = source.time_base
        params.setdefault('dtype', 'int16')
        params.setdefault('n_channels_dat', len(source.arrays.get('channel_map.npy', [])))
        (temporary / 'params.py').write_text(
            ''.join(f'{key} = {value!r}\n' for key, value in params.items()), encoding='utf-8'
        )
        summaries = build_summaries(source, temporary, bin_seconds, max_bytes=max_bytes)
        final_windows = []
        for start, end in windows:
            contributors = [
                record for record in plan if record['start'] < end and record['end'] > start
            ]
            kinds = sorted({record['kind'] for record in contributors})
            final_windows.append(
                {
                    'start': start,
                    'end': end,
                    'segment': 0,
                    'block': contributors[0]['block'],
                    'kind': kinds[0] if len(kinds) == 1 else 'mixed',
                    'kinds': kinds,
                    'strata': sorted({record['stratum'] for record in contributors}),
                    'exposure_seconds': (end - start) / source.sample_rate,
                }
            )
        manifest = {
            'format': 'phy_bounded_curation',
            'format_version': 1,
            'source': {
                'identity': source.identity,
                'revision': source.revision,
                'path': str(source.path),
            },
            'sample_rate': source.sample_rate,
            'num_samples': source.num_samples,
            'duration_seconds': source.num_samples / source.sample_rate,
            'valid_intervals': source.valid_intervals.tolist(),
            'num_spikes': source.n_spikes,
            'bounded_num_spikes': n_bounded,
            'unit_ids': source.unit_ids.tolist(),
            'counts': source.counts.tolist(),
            'bounded_counts': [bounded_counts[int(unit)] for unit in source.unit_ids],
            'waveform_counts': [waveform_counts[int(unit)] for unit in source.unit_ids],
            'waveform_coverage': {
                'source_rows': len(source.arrays['_phy_spikes_subset.spikes.npy'])
                if source.arrays.keys() >= WAVEFORMS
                else 0,
                'retained_rows': sum(waveform_counts.values()),
                'dropped_rows': (
                    len(source.arrays['_phy_spikes_subset.spikes.npy'])
                    if source.arrays.keys() >= WAVEFORMS
                    else 0
                )
                - sum(waveform_counts.values()),
            },
            'producer_window_plan': producer_provenance,
            'producer_window_estimate': producer_estimate,
            'next_cluster_id': source.next_cluster_id,
            'global_spike_ids_file': '_phy_spike_global_ids.npy',
            'base_assignments_file': '_phy_spike_clusters_base.npy',
            'windows_file': '_phy_curation_windows.npy',
            'windows': final_windows,
            'accepted_window_candidates': plan,
            'omitted_windows': omitted,
            'metadata': metadata,
            'base_metadata': metadata,
            'metadata_schema': source.metadata_schema,
            'metadata_policy': POLICY,
            'metadata_provenance': source.metadata_provenance,
            'summaries': summaries,
            **{key: value for key, value in summaries.items() if key.endswith('_file')},
            'selection': {
                'algorithm': 'shared_windows_v1',
                'window_policy': window_policy,
                'resolved_window_policy': resolved_policy,
                'seed': int(seed),
                'window_seconds': window_seconds,
                'stratum_seconds': stratum_seconds,
            },
            'budgets': {
                'max_spikes': int(max_spikes),
                'max_bytes': int(max_bytes),
                'batch_size': int(batch_size),
                'summary_bytes': int(summary_bytes),
                'estimated_bytes': int(
                    fixed_bytes
                    + n_bounded * row_bytes
                    + _selected_waveform_count(source, windows) * waveform_bytes
                    + (len(windows) + len(plan) + len(omitted)) * 256
                ),
            },
            'capabilities': {
                'merge': True,
                'label': True,
                'split': False,
                'full_attributes': sorted(scalar_attributes(source)),
                'predicate_split_attributes': (
                    ['amplitudes.npy'] if 'amplitude' in scalar_attributes(source) else []
                ),
                'stored_waveforms': bool(sum(waveform_counts.values())),
                'raw_traces': False,
                'normalized_correlograms': False,
            },
            'source_array_inventory': {
                name: {
                    'role': 'per_spike'
                    if name in PER_SPIKE
                    else 'stored_waveform'
                    if name in WAVEFORMS
                    else 'waveform_sidecar'
                    if name in waveform_sidecars
                    else 'static'
                    if name in STATIC
                    else 'excluded_window_sidecar',
                    'shape': list(array.shape),
                    'dtype': str(array.dtype),
                    'bytes': int(array.nbytes),
                }
                for name, array in source.arrays.items()
            },
            'excluded_files': sorted(
                p.name
                for p in source.path.iterdir()
                if p.name not in (PER_SPIKE | STATIC | WAVEFORMS | waveform_sidecars)
                and p.name
                not in (
                    'params.py',
                    '_phy_cluster_metadata.json',
                    SCHEMA_FILE,
                    'curation.json',
                    'curation_review.json',
                )
                and (not p.name.startswith('cluster_') or p.stem == 'cluster_info')
            ),
        }
        if getattr(source, 'time_base', None) is not None:
            manifest['time_base'] = source.time_base
        manifest['retired_units'] = source.retired_units
        manifest['proposition_documents'] = {}
        for document in producer_documents:
            # Root review is a convenience copy; its inherited baseline is immutable.
            name = (
                '_phy_curation_review_base.json'
                if document.name == 'curation_review.json'
                else document.name
            )
            shutil.copyfile(document, temporary / name)
            manifest['proposition_documents'][document.name] = {
                'filename': name,
                'sha256': producer_hashes[document.name],
            }
            if hash_file(temporary / name) != producer_hashes[document.name]:
                raise ValueError('Producer proposition document changed during preparation.')
            if name != document.name:
                shutil.copyfile(document, temporary / document.name)
        bound_paths = [
            *temporary.glob('*.npy'),
            temporary / 'params.py',
            temporary / '_phy_cluster_metadata.json',
            temporary / SCHEMA_FILE,
            *(temporary / table['filename'] for table in source.metadata_schema['tables']),
            *(temporary / item['filename'] for item in manifest['proposition_documents'].values()),
        ]
        manifest['bounded_files'] = {p.name: hash_file(p) for p in sorted(bound_paths)}
        manifest['preparation_identity'] = hash_json(manifest)
        (temporary / '_phy_curation_manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        actual_bytes = sum(p.stat().st_size for p in temporary.iterdir())
        if actual_bytes > max_bytes:
            raise ValueError(f'Final output exceeds max_bytes: {actual_bytes} bytes.')
        for array in (global_ids, base, *destinations.values()):
            close_array(array)
        for path in temporary.iterdir():
            if path.is_file():
                # Windows _commit (os.fsync) requires a writable descriptor.
                with path.open('r+b') as stream:
                    os.fsync(stream.fileno())
        _fsync_directory(temporary)
        if output_dir.exists():
            raise ValueError(f'Output already exists: {output_dir}.')
        from .curation import BoundedSession

        checked = BoundedSession(temporary)
        checked.close()
        publish_directory(temporary, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        for array in (global_ids, base, *destinations.values()):
            close_array(array)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def _fsync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _copy_waveforms(source, output, global_ids, row_bounds):
    """Intersect sparse row IDs with selected spans in bounded working memory."""
    upstream_ids = source.arrays['_phy_spikes_subset.spikes.npy']
    spans = [
        (int(np.searchsorted(upstream_ids, first)), int(np.searchsorted(upstream_ids, stop)))
        for first, stop in row_bounds
    ]
    n = sum(stop - first for first, stop in spans)
    targets = {}
    waveform_names = WAVEFORMS | (source.arrays.keys() & {'_phy_spikes_subset.sample_index.npy'})
    try:
        for name in waveform_names:
            original = source.arrays[name]
            targets[name] = np.lib.format.open_memmap(
                output / name, mode='w+', dtype=original.dtype, shape=(n, *original.shape[1:])
            )
        cursor = 0
        for first, stop in spans:
            for start in range(first, stop, source.batch_size):
                end = min(start + source.batch_size, stop)
                length = end - start
                targets['_phy_spikes_subset.spikes.npy'][cursor : cursor + length] = (
                    np.searchsorted(global_ids, upstream_ids[start:end])
                )
                for name in waveform_names - {'_phy_spikes_subset.spikes.npy'}:
                    targets[name][cursor : cursor + length] = source.arrays[name][start:end]
                cursor += length
        for array in targets.values():
            array.flush()
    finally:
        for array in targets.values():
            close_array(array)
