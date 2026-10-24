"""Stream semantic curation into a new complete sorting, never the source."""

import csv
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .curation import (
    BoundedSession,
    _digest,
    _fsync_directory,
    _write_json,
    interpret_events,
    project_assignments,
)
from .source import PER_SPIKE, Source, _has_raw_paths, close_array, publish_directory


def _merge_mapping(unit_ids, events):
    """Resolve a whole-unit merge journal once, independently of spike count."""
    lineage = {int(unit): [int(unit)] for unit in unit_ids}
    for event in events:
        if event['operation'] == 'merge':
            members = []
            for parent in event['parents']:
                members.extend(lineage.pop(parent))
            lineage[event['target']] = members
    mapping = {old: new for new, members in lineage.items() for old in members}
    keys = np.array(sorted(mapping), dtype=np.int64)
    targets = np.array([mapping[int(key)] for key in keys], dtype=np.int64)
    return keys, targets


def _write_metadata(path, metadata):
    for field, values in sorted(metadata.items()):
        if not field or any(char in field for char in ('/', '\\', '\0')):
            raise ValueError('Metadata fields must be safe local filenames.')
        with (path / f'cluster_{field}.tsv').open('w', encoding='utf8', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['cluster_id', field])
            for cluster, value in sorted(values.items()):
                if value is not None:
                    writer.writerow([cluster, value])
            stream.flush()
            os.fsync(stream.fileno())
    _write_json(path / '_phy_cluster_metadata.json', metadata)


def apply_curation(source_params, bounded_dir, output_dir, batch_size=100000):
    """Validate identity/revision and replay only the last committed generation.

    The output is published as one new directory after validation. Full source
    arrays remain file-backed; only one assignment batch and the unit map reside
    in memory. Arbitrary selected-row splits have no journal interpretation.
    """
    session = BoundedSession(bounded_dir)
    source = None
    try:
        if session.checkpoint_path is None:
            raise ValueError('Save a bounded semantic checkpoint before applying curation.')
        manifest = session.manifest
        source = Source(
            source_params,
            duration_seconds=manifest['num_samples'] / manifest['sample_rate'],
            valid_intervals=manifest['valid_intervals'],
            batch_size=batch_size,
        )
        return _apply_curation(source, session, output_dir, batch_size)
    finally:
        if source is not None:
            source.close()
        session.close()


def _apply_curation(source, session, output_dir, batch_size):
    manifest = session.manifest
    expected = manifest['source']
    if source.identity != expected['identity']:
        raise ValueError('Source coordinate/feature identity does not match bounded preparation.')
    if source.revision != expected['revision']:
        raise ValueError('Source assignment revision does not match journal parent revision.')
    if (
        source.unit_ids.tolist() != manifest['unit_ids']
        or source.n_spikes != manifest['num_spikes']
    ):
        raise ValueError('Source catalog or spike count does not match bounded preparation.')
    # Identity/revision bind the complete source. Validate the actual projection
    # too, so a tampered bounded base cannot silently replay unrelated evidence.
    source_assignments = source.arrays['spike_clusters.npy'].reshape(-1)
    for start in range(0, len(session.global_ids), batch_size):
        rows = session.global_ids[start : start + batch_size]
        if not np.array_equal(
            source_assignments[rows],
            session.base_assignments[start : start + batch_size],
        ):
            raise ValueError('Immutable bounded assignments do not match source rows.')
    for name in sorted(source.arrays.keys() & (PER_SPIKE - {'spike_clusters.npy'})):
        bounded = np.load(session.path / name, mmap_mode='r', allow_pickle=False)
        try:
            if bounded.shape != (len(session.global_ids), *source.arrays[name].shape[1:]):
                raise ValueError(f'Bounded attribute has invalid source alignment: {name}.')
            for start in range(0, len(session.global_ids), batch_size):
                rows = session.global_ids[start : start + batch_size]
                if not np.array_equal(
                    source.arrays[name][rows],
                    bounded[start : start + batch_size],
                    equal_nan=True,
                ):
                    raise ValueError(f'Bounded attribute does not match source rows: {name}.')
        finally:
            close_array(bounded)
    active_units, metadata, _ = interpret_events(source.unit_ids, source.metadata, session.events)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Curated output already exists: {output}.')
    if output == source.path or source.path in output.parents:
        raise ValueError('Curated output must be outside the immutable source directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = sum(
        (source.path / name).stat().st_size
        for name in source.arrays
        if name != 'spike_clusters.npy'
    )
    estimated_bytes += source.n_spikes * 8 + 1024 * 1024
    free = shutil.disk_usage(output.parent).free
    if free < estimated_bytes:
        raise OSError(
            f'Insufficient disk space for full replay: estimated at least {estimated_bytes:,} bytes, {free:,} available.'
        )
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    curated = None
    assignments = None
    try:
        # Copy only validated canonical assets, not GUI caches or producer reviews.
        for name in source.arrays:
            if name != 'spike_clusters.npy':
                shutil.copyfile(source.path / name, stage / name)
        shutil.copyfile(source.params_path, stage / 'params.py')
        upstream_manifest = source.path / '_phy_spikes_subset.manifest.json'
        if upstream_manifest.exists():
            shutil.copyfile(upstream_manifest, stage / upstream_manifest.name)
        if source.time_base is not None and source.timestamps_path.exists():
            shutil.copyfile(source.timestamps_path, stage / 'timestamps.json')
        # Preserve source params semantics and make external raw paths absolute.
        # The full export can contain no raw traces; immutable array rows suffice.
        dat_path = source.params.get('dat_path')
        if _has_raw_paths(dat_path):
            raw_paths = dat_path if isinstance(dat_path, (list, tuple)) else [dat_path]
            resolved = [
                str((source.path / raw).resolve()) if _has_raw_paths(raw) else raw
                for raw in raw_paths
            ]
            dat_path = resolved if isinstance(dat_path, (list, tuple)) else resolved[0]
        with (stage / 'params.py').open('a', encoding='utf8') as stream:
            stream.write(f'\nnum_samples = {source.num_samples!r}\n')
            stream.write(f'valid_intervals = {source.valid_intervals.tolist()!r}\n')
            stream.write(f'dat_path = {dat_path!r}\n')
        assignments = np.lib.format.open_memmap(
            stage / 'spike_clusters.npy',
            mode='w+',
            dtype=np.int64,
            shape=source.arrays['spike_clusters.npy'].shape,
        )
        flat = assignments.reshape(-1)
        has_split = any(event['operation'] == 'threshold_split' for event in session.events)
        keys, targets = (
            (None, None) if has_split else _merge_mapping(source.unit_ids, session.events)
        )
        for batch in source.iter_batches():
            values = batch['spike_clusters.npy']
            if has_split:
                projected = project_assignments(values, session.events, batch)
            else:
                positions = np.searchsorted(keys, values)
                projected = targets[positions]
            flat[batch['start'] : batch['stop']] = projected
        assignments.flush()
        assignments._mmap.close()
        del flat, assignments
        assignments = None
        _write_metadata(stage, metadata)
        _write_json(stage / '_phy_curation_journal.json', session.events)
        # Re-reading verifies alignment, full catalog and immutable byte digests.
        curated = Source(stage / 'params.py', batch_size=batch_size)
        if curated.identity != source.identity or curated.n_spikes != source.n_spikes:
            raise ValueError('Curated output changed immutable source coordinates.')
        if curated.metadata != metadata:
            raise ValueError('Curated output metadata failed exact round-trip validation.')
        actual = curated.arrays['spike_clusters.npy'].reshape(-1)
        for batch in source.iter_batches():
            if has_split:
                expected_batch = project_assignments(
                    batch['spike_clusters.npy'], session.events, batch
                )
            else:
                expected_batch = targets[np.searchsorted(keys, batch['spike_clusters.npy'])]
            if not np.array_equal(actual[batch['start'] : batch['stop']], expected_batch):
                raise ValueError('Curated output assignments failed semantic replay verification.')
        _write_json(
            stage / '_phy_curation_lineage.json',
            {
                'format': 'phy_curation_lineage',
                'format_version': 1,
                'coordinate_identity': source.identity,
                'parent_revision': source.revision,
                'journal_sha256': _digest(session.events),
                'revision': curated.revision,
                'preparation_sha256': _digest(manifest),
                'next_cluster_id': session.next_cluster_id,
                'unit_ids': sorted(active_units),
            },
        )
        curated.close()
        for path in stage.iterdir():
            if path.is_file():
                # Windows _commit (os.fsync) requires a writable descriptor.
                with path.open('r+b') as stream:
                    os.fsync(stream.fileno())
        _fsync_directory(stage)
        # Refuse replacement even when a racing caller creates the target.
        if output.exists():
            raise FileExistsError(f'Curated output already exists: {output}.')
        publish_directory(stage, output)
        _fsync_directory(output.parent)
    except BaseException:
        if curated is not None:
            curated.close()
        if assignments is not None:
            assignments._mmap.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output
