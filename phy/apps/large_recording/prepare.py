"""Publish a self-contained bounded evidence package from a full sorting."""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

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
    for start, end in sorted(windows):
        if result and (
            start < result[-1][1] or (start == result[-1][1] and start not in block_starts)
        ):
            result[-1][1] = max(result[-1][1], int(end))
        else:
            result.append([int(start), int(end)])
    return result


def _write_metadata(path, metadata):
    for field, values in metadata.items():
        if (
            not isinstance(field, str)
            or not field
            or '/' in field
            or '\\' in field
            or field in ('.', '..')
        ):
            raise ValueError('Metadata field names must be safe filenames.')
        with (path / f'cluster_{field}.tsv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['cluster_id', field])
            writer.writerows(sorted(values.items()))


def _selected_waveform_count(source, windows):
    ids = source.arrays.get('_phy_spikes_subset.spikes.npy')
    if ids is None:
        return 0
    return sum(
        int(np.searchsorted(ids, stop) - np.searchsorted(ids, start))
        for start, stop in _row_bounds(source.arrays['spike_times.npy'].reshape(-1), windows)
    )


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
):
    """Choose complete shared windows, write bounded arrays, and publish atomically.

    Survey windows use a seeded offset within each acquisition stratum. A second
    pass tries one first-spike rescue window for every unit without evidence.
    Candidate windows that exceed either budget are explicitly omitted; spikes
    within accepted windows are never independently thinned.
    """
    with Source(params_path, duration_seconds, valid_intervals, batch_size) as source:
        return _prepare_source(
            source,
            output_dir,
            max_spikes,
            max_bytes,
            window_seconds,
            stratum_seconds,
            seed,
            batch_size,
            bin_seconds,
        )


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
):
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
    fixed_bytes = (
        sum(path.stat().st_size for path in producer_documents)
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
    row_bytes = 16  # int64 global row mapping and baseline assignment
    for name in source.arrays.keys() & PER_SPIKE:
        array = source.arrays[name]
        row_bytes += array.dtype.itemsize * int(np.prod(array.shape[1:], dtype=np.int64))
    waveform_bytes = 0
    if source.arrays.keys() >= WAVEFORMS:
        for name in WAVEFORMS:
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
        size = fixed_bytes + n * row_bytes + n_waveforms * waveform_bytes + len(candidate) * 256
        record = {
            'start': int(start),
            'end': int(end),
            'kind': kind,
            'segment': 0,
            'block': int(segment),
            'stratum': int(stratum_id),
        }
        if n > max_spikes or size > max_bytes:
            omitted.append(
                {**record, 'reason': 'spike_budget' if n > max_spikes else 'byte_budget'}
            )
            return False
        windows[:] = candidate
        plan.append(record)
        return True

    # Candidate enumeration is scalar and bounded by output-window metadata,
    # rather than constructing recording-length arrays.
    for segment, (first, last) in enumerate(source.valid_intervals):
        for start in range(int(first), int(last), stratum):
            end = min(start + stratum, int(last))
            length = min(width, end - start)
            offset = int(rng.integers(0, end - start - length + 1))
            accept(start + offset, start + offset + length, 'survey', segment, start // stratum)
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
        + len(windows) * 256
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
        _write_metadata(temporary, source.metadata)
        (temporary / '_phy_cluster_metadata.json').write_text(
            json.dumps(source.metadata, sort_keys=True) + '\n', encoding='utf-8'
        )
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
            'next_cluster_id': source.next_cluster_id,
            'global_spike_ids_file': '_phy_spike_global_ids.npy',
            'base_assignments_file': '_phy_spike_clusters_base.npy',
            'windows_file': '_phy_curation_windows.npy',
            'windows': final_windows,
            'accepted_window_candidates': plan,
            'omitted_windows': omitted,
            'metadata': source.metadata,
            'base_metadata': source.metadata,
            'summaries': summaries,
            **{key: value for key, value in summaries.items() if key.endswith('_file')},
            'selection': {
                'algorithm': 'shared_windows_v1',
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
                    + len(windows) * 256
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
                if p.name not in (PER_SPIKE | STATIC | WAVEFORMS)
                and p.name
                not in (
                    'params.py',
                    '_phy_cluster_metadata.json',
                    'curation.json',
                    'curation_review.json',
                )
                and (not p.name.startswith('cluster_') or p.stem == 'cluster_info')
            ),
        }
        for document in producer_documents:
            shutil.copyfile(document, temporary / document.name)
        bound_paths = [
            *temporary.glob('*.npy'),
            temporary / 'params.py',
            temporary / '_phy_cluster_metadata.json',
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
    try:
        for name in WAVEFORMS:
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
                for name in WAVEFORMS - {'_phy_spikes_subset.spikes.npy'}:
                    targets[name][cursor : cursor + length] = source.arrays[name][start:end]
                cursor += length
        for array in targets.values():
            array.flush()
    finally:
        for array in targets.values():
            close_array(array)
