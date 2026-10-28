"""Bounded offline sorting and authenticated block-tree construction."""

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .curation import _fsync_directory
from .range_files import PAGE_BYTES, authenticate_file
from .range_intervals import RangeUnavailableError
from .range_math import add_stats, node_dtype, row_stats
from .source import hash_json, publish_directory
from .summary_store import (
    _ANY_REFERENCE,
    SummaryStore,
    SummaryStorePublishedError,
    _atomic_json,
    summary_store_reference,
)

FORMAT = 'phy_amplitude_range_index'
VERSION = 1
DEFAULT_DIRECTORY = 'amplitude-index'


def layout(engine):
    inventory = engine.manifest['source_array_inventory']
    if 'amplitude' not in engine.fields:
        raise RangeUnavailableError('Scalar amplitudes are required for range indexing.')
    names = [('amplitude', np.dtype(inventory['amplitudes.npy']['dtype']))]
    if names[0][1].kind not in 'fiu' or names[0][1].itemsize > 8:
        raise RangeUnavailableError('This amplitude dtype uses the verified row fallback.')
    if 'depth' in engine.fields:
        name = 'spike_depths.npy' if 'spike_depths.npy' in inventory else 'spike_positions.npy'
        dtype = np.dtype(inventory[name]['dtype'])
        if dtype.kind not in 'fiu' or dtype.itemsize > 8:
            raise RangeUnavailableError('This depth dtype uses the verified row fallback.')
        names.append(('depth', dtype))
    payload = np.dtype(names)
    directory = np.dtype(
        [
            ('start', '<i8'),
            ('finite', '<i8'),
            ('total', '<i8'),
            ('keys', '<i8'),
            ('tree', '<i8'),
            ('leaves', '<i8'),
            ('minimum', names[0][1]),
            ('maximum', names[0][1]),
            *[
                (f'nf_depth_{kind}', '<i8' if kind == 'count' else '<f8')
                for kind in ('sum', 'count', 'abs', 'error')
                if 'depth' in engine.fields
            ],
        ]
    )
    return payload, directory, node_dtype(engine.fields)


@dataclass(frozen=True)
class RangeEstimate:
    final_bytes: int
    peak_extra_bytes: int
    payload_bytes: int


def estimate_range_index(engine, block_rows=128, units=None):
    payload, directory, node = layout(engine)
    if type(block_rows) is not int or not 1 <= block_rows <= 65536:
        raise ValueError('block_rows must be an integer between 1 and 65536.')
    counts = dict(zip(engine.manifest['unit_ids'], engine.manifest['counts']))
    units = list(counts) if units is None else list(units)
    if not set(units) <= counts.keys() or len(set(units)) != len(units):
        raise ValueError('Unknown or repeated source units.')
    n = sum(int(counts[u]) for u in units)
    segments = min(n, len(units) * engine.bins)
    # Padding each nonempty segment to a power of two costs <4 nodes/block.
    blocks = (n + block_rows - 1) // block_rows + segments
    data = n * payload.itemsize
    meta = blocks * (4 * node.itemsize + payload['amplitude'].itemsize) + segments * 64
    dirs = len(counts) * engine.bins * directory.itemsize
    raw = data + meta + dirs
    # Merkle trees use <4*32 bytes/page, plus their small minimum allocations.
    final = raw + math.ceil(raw / PAGE_BYTES) * 128 + 1048576
    scratch = 2 * max((int(counts[u]) for u in units), default=0) * payload.itemsize
    largest = max((int(counts[u]) for u in units), default=0)
    largest_blocks = max(1, (largest + block_rows - 1) // block_rows)
    scratch += 2 * (1 << (largest_blocks - 1).bit_length()) * node.itemsize
    scratch += (max(data, meta, dirs) // PAGE_BYTES + 1) * 32
    return RangeEstimate(final, final + scratch, data)


def preflight_summary_store(package_path, max_bytes):
    """Check the combined metadata estimate before expensive row preparation."""
    from .live_summaries import LiveSummaryEngine
    from .summary_store import estimate_summary_store

    with LiveSummaryEngine(package_path) as engine:
        try:
            estimate = estimate_range_index(engine).final_bytes
        except RangeUnavailableError:
            return  # Unsupported scalar layouts keep verified rows.
        estimate += estimate_summary_store(package_path).estimated_bytes
        if estimate > max_bytes:
            raise ValueError(
                f'Combined store estimate {estimate:,} exceeds max_bytes; '
                'increase the store budget or use build-summary-store --rows-only.'
            )


class _SortedRuns:
    """Stable bounded runs and vectorized pairwise merging, never row-wise heap merge."""

    def __init__(self, directory, dtype, memory_bytes, cancel):
        self.directory, self.dtype = directory, dtype
        # Reserve stable-sort permutations/workspace even for one-byte scalars.
        self.limit = max(
            dtype.itemsize, memory_bytes // (4 * dtype.itemsize + 16) * dtype.itemsize
        )
        self.parts, self.bytes, self.runs = [], 0, []
        self.cancel = cancel
        self.merge_rows = max(1, memory_bytes // (8 * dtype.itemsize + 96))

    def add(self, records):
        rows = max(1, self.limit // self.dtype.itemsize)
        for first in range(0, len(records), rows):
            part = records[first : first + rows].copy()
            if self.parts and self.bytes + part.nbytes > self.limit:
                self._flush()
            self.parts.append(part)
            self.bytes += part.nbytes
            if self.bytes >= self.limit:
                self._flush()

    def _flush(self):
        from .live_summaries import _check_cancel

        _check_cancel(self.cancel)
        values = np.concatenate(self.parts)
        values = values[np.argsort(values['amplitude'], kind='stable')]
        handle, filename = tempfile.mkstemp(prefix='sorted-', dir=self.directory)
        with os.fdopen(handle, 'wb') as stream:
            stream.write(values.tobytes())
        self.runs.append(Path(filename))
        self.parts.clear()
        self.bytes = 0

    def _merge(self, left_path, right_path):
        from .live_summaries import _check_cancel

        handle, filename = tempfile.mkstemp(prefix='merged-', dir=self.directory)
        with (
            os.fdopen(handle, 'wb') as output,
            left_path.open('rb') as left,
            right_path.open('rb') as right,
        ):
            a = b = np.empty(0, dtype=self.dtype)
            while True:
                _check_cancel(self.cancel)
                if not len(a):
                    a = np.frombuffer(
                        left.read(self.merge_rows * self.dtype.itemsize), dtype=self.dtype
                    )
                if not len(b):
                    b = np.frombuffer(
                        right.read(self.merge_rows * self.dtype.itemsize), dtype=self.dtype
                    )
                if not len(a) and not len(b):
                    break
                if not len(a):
                    output.write(b.tobytes())
                    b = b[:0]
                    continue
                if not len(b):
                    output.write(a.tobytes())
                    a = a[:0]
                    continue
                bound = min(a['amplitude'][-1], b['amplitude'][-1])
                na = int(np.searchsorted(a['amplitude'], bound, side='right'))
                # If the left chunk ends at the boundary, future left rows
                # may have the same key. Keep right ties until those are read.
                side = 'left' if bound == a['amplitude'][-1] else 'right'
                nb = int(np.searchsorted(b['amplitude'], bound, side=side))
                aa, bb = a[:na], b[:nb]
                if not na:
                    result = bb
                elif not nb:
                    result = aa
                else:
                    result = np.empty(na + nb, dtype=self.dtype)
                    positions_a = np.arange(na) + np.searchsorted(
                        bb['amplitude'], aa['amplitude'], side='left'
                    )
                    positions_b = np.arange(nb) + np.searchsorted(
                        aa['amplitude'], bb['amplitude'], side='right'
                    )
                    result[positions_a] = aa
                    result[positions_b] = bb
                output.write(result.tobytes())
                a, b = a[na:], b[nb:]
        left_path.unlink()
        right_path.unlink()
        return Path(filename)

    def finish(self):
        if not self.runs:
            values = np.concatenate(self.parts) if self.parts else np.empty(0, dtype=self.dtype)
            self.parts.clear()
            return values[np.argsort(values['amplitude'], kind='stable')], None
        if self.parts:
            self._flush()
        while len(self.runs) > 1:
            output = []
            for i in range(0, len(self.runs), 2):
                output.append(
                    self._merge(self.runs[i], self.runs[i + 1])
                    if i + 1 < len(self.runs)
                    else self.runs[i]
                )
            self.runs = output
        path = self.runs.pop()
        return None, path


def _align(stream, width=64):
    count = (-stream.tell()) % width
    if count:
        stream.write(b'\0' * count)


def _write_segment(
    sorter,
    total,
    nonfinite,
    values_file,
    meta_file,
    fields,
    block_rows,
    memory_bytes,
    directory_dtype,
    node,
    cancel,
):
    from .live_summaries import _check_cancel

    records, run = sorter.finish()
    n = len(records) if records is not None else run.stat().st_size // sorter.dtype.itemsize
    entry = np.zeros(1, dtype=directory_dtype)
    entry['start'], entry['finite'], entry['total'] = values_file.tell(), n, total
    if 'depth' in fields:
        for kind in ('sum', 'count', 'abs', 'error'):
            entry['nf_depth_' + kind] = nonfinite['depth_' + kind]
    if not n:
        return entry
    blocks = (n + block_rows - 1) // block_rows
    leaves = 1 << (blocks - 1).bit_length()
    entry['leaves'] = leaves
    _align(meta_file)
    entry['keys'] = meta_file.tell()
    handle, tree_name = tempfile.mkstemp(prefix='aggregate-', dir=sorter.directory)
    tree_path = Path(tree_name)
    stream = None
    try:
        with os.fdopen(handle, 'w+b') as tree:
            stream = None if records is not None else run.open('rb')
            tree.truncate(2 * leaves * node.itemsize)
            tree.seek(leaves * node.itemsize)
            batch_blocks = max(
                1,
                min(
                    8192, memory_bytes // (block_rows * (sorter.dtype.itemsize + 32 * len(fields)))
                ),
            )
            first, last = None, None
            for start in range(0, n, batch_blocks * block_rows):
                _check_cancel(cancel)
                stop = min(n, start + batch_blocks * block_rows)
                chunk = (
                    records[start:stop]
                    if records is not None
                    else np.frombuffer(
                        stream.read((stop - start) * sorter.dtype.itemsize), dtype=sorter.dtype
                    )
                )
                if first is None:
                    first = chunk['amplitude'][0]
                last = chunk['amplitude'][-1]
                values_file.write(chunk.tobytes())
                meta_file.write(chunk['amplitude'][::block_rows].tobytes())
                full = len(chunk) // block_rows * block_rows
                if full:
                    stats = row_stats(chunk[:full].reshape(-1, block_rows), fields, node)
                    tree.write(stats.tobytes())
                if full < len(chunk):
                    tree.write(row_stats(chunk[full:], fields, node).tobytes())
            entry['minimum'], entry['maximum'] = first, last
            # Unused leaves are known zeros; write in bounded chunks, including Windows.
            remaining = (leaves - blocks) * node.itemsize
            zero = bytes(min(1048576, remaining))
            while remaining:
                _check_cancel(cancel)
                part = min(remaining, len(zero))
                tree.write(zero[:part])
                remaining -= part
            tree.flush()
            width = leaves // 2
            while width:
                for first_node in range(width, 2 * width, 8192):
                    _check_cancel(cancel)
                    count = min(8192, 2 * width - first_node)
                    tree.seek(2 * first_node * node.itemsize)
                    children = np.frombuffer(tree.read(2 * count * node.itemsize), dtype=node)
                    parents = add_stats(children[::2], children[1::2], fields)
                    tree.seek(first_node * node.itemsize)
                    tree.write(parents.tobytes())
                tree.flush()
                width //= 2
            _align(meta_file)
            entry['tree'] = meta_file.tell()
            tree.seek(0)
            while raw := tree.read(1048576):
                _check_cancel(cancel)
                meta_file.write(raw)
    finally:
        if stream is not None:
            stream.close()
        if run is not None:
            run.unlink(missing_ok=True)
        tree_path.unlink(missing_ok=True)
    return entry


def build_amplitude_index(
    package_path,
    output_dir,
    block_rows=128,
    max_bytes=68719476736,
    memory_bytes=67108864,
    units=None,
    cancel=None,
    progress=None,
    before_publish=None,
    attach=True,
    expected_reference=_ANY_REFERENCE,
):
    """Prepare an optional sorted range accelerator from a verified row snapshot."""
    from .live_summaries import LiveSummaryEngine, _check_cancel

    package, output = Path(package_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f'Amplitude index already exists: {output}.')
    if output == package or package in output.parents:
        raise ValueError('Amplitude index must be outside the bounded package.')
    if memory_bytes < 1048576 or max_bytes <= 0:
        raise ValueError('Index memory must be at least 1 MiB and disk budget must be positive.')
    staged = None
    initial_reference = summary_store_reference(package, allow_invalid=True)
    if expected_reference is not _ANY_REFERENCE and initial_reference != expected_reference:
        raise ValueError('The attached summary store changed before amplitude-index preparation.')
    with LiveSummaryEngine(package) as engine, SummaryStore(engine) as store:
        if output == engine.source_dir.resolve() or engine.source_dir.resolve() in output.parents:
            raise ValueError('Amplitude index must be outside the original source.')
        estimate = estimate_range_index(engine, block_rows, units)
        row_bytes = sum((store.path / name).stat().st_size for name in ('data.bin', 'index.json'))
        if row_bytes + estimate.final_bytes > max_bytes:
            raise ValueError('Combined row store and amplitude index exceed max_bytes.')
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < estimate.peak_extra_bytes:
            raise OSError('Insufficient free space for amplitude index and sorting scratch.')
        payload, directory_dtype, node = layout(engine)
        counts = dict(zip(engine.manifest['unit_ids'], engine.manifest['counts']))
        selected = set(counts if units is None else units)
        staged = Path(tempfile.mkdtemp(prefix=f'.{output.name}.building-', dir=output.parent))
        scratch = staged / 'sorting'
        scratch.mkdir()
        completed = 0
        total_rows = sum(int(counts[u]) for u in selected)
        try:
            with (
                (staged / 'values.bin').open('wb') as values_file,
                (staged / 'nodes.bin').open('wb') as meta_file,
                (staged / 'segments.bin').open('w+b') as directories,
            ):
                directories.truncate(len(counts) * engine.bins * directory_dtype.itemsize)
                for ordinal, unit in enumerate(engine.manifest['unit_ids']):
                    if unit not in selected:
                        continue
                    current, sorter, total = None, None, 0
                    nonfinite = np.zeros(1, dtype=node)

                    def finish(current, sorter, total, nonfinite, ordinal):
                        if current is None:
                            return
                        entry = _write_segment(
                            sorter,
                            total,
                            nonfinite[0],
                            values_file,
                            meta_file,
                            engine.fields,
                            block_rows,
                            memory_bytes,
                            directory_dtype,
                            node,
                            cancel,
                        )
                        directories.seek(
                            (ordinal * engine.bins + current) * directory_dtype.itemsize
                        )
                        directories.write(entry.tobytes())

                    batch_rows = max(
                        1, min(500000, memory_bytes // (8 * store.dtype.itemsize + 32))
                    )
                    for rows in store.batches(unit, batch_rows, cancel):
                        bin_ids = rows['bin']
                        if np.any(bin_ids >= engine.bins) or np.any(bin_ids[1:] < bin_ids[:-1]):
                            raise RangeUnavailableError(
                                'Row snapshot bins are invalid or unordered.'
                            )
                        boundaries = np.r_[0, np.flatnonzero(np.diff(bin_ids)) + 1, len(rows)]
                        for first, last in zip(boundaries[:-1], boundaries[1:]):
                            _check_cancel(cancel)
                            bin_id = int(bin_ids[first])
                            if current is not None and bin_id < current:
                                raise RangeUnavailableError('Row snapshot bin order changed.')
                            if bin_id != current:
                                finish(current, sorter, total, nonfinite, ordinal)
                                current, total = bin_id, 0
                                sorter = _SortedRuns(scratch, payload, memory_bytes, cancel)
                                nonfinite = np.zeros(1, dtype=node)
                            chunk = rows[first:last]
                            finite = np.isfinite(chunk['amplitude'])
                            total += len(chunk)
                            if 'depth' in engine.fields and np.any(~finite):
                                packed = np.empty(np.count_nonzero(~finite), dtype=payload)
                                for field in engine.fields:
                                    packed[field] = chunk[field][~finite]
                                nonfinite = add_stats(
                                    nonfinite,
                                    row_stats(packed, engine.fields, node),
                                    engine.fields,
                                )
                            packed = np.empty(np.count_nonzero(finite), dtype=payload)
                            for field in engine.fields:
                                packed[field] = chunk[field][finite]
                            sorter.add(packed)
                        completed += len(rows)
                        if progress:
                            progress('sorting amplitude ranges', completed, total_rows)
                    finish(current, sorter, total, nonfinite, ordinal)
                for stream_ in (values_file, meta_file, directories):
                    stream_.flush()
                    os.fsync(stream_.fileno())
            scratch.rmdir()
            files = {
                name: authenticate_file(staged / f'{name}.bin', cancel, progress)
                for name in ('values', 'nodes', 'segments')
            }
            catalog = {
                'format': FORMAT,
                'version': VERSION,
                'arithmetic_version': 1,
                'binding_sha256': engine.binding_digest,
                'parent_sha256': hash_json(store.index),
                'parent_path': str(store.path),
                'block_rows': block_rows,
                'indexed_units': sorted(selected),
                'files': files,
            }
            _atomic_json(staged / 'range.json', catalog)
            actual = sum(p.stat().st_size for p in staged.iterdir())
            if actual + row_bytes > max_bytes:
                raise ValueError('Combined index exceeds max_bytes after construction.')
            _check_cancel(cancel)
            if before_publish:
                before_publish()
            publish_directory(staged, output)
            staged = None
            try:
                _fsync_directory(output.parent)
                if attach:
                    from .range_index import attach_amplitude_index

                    attach_amplitude_index(package, output, expected_reference=initial_reference)
            except (ValueError, OSError) as error:
                raise SummaryStorePublishedError(output, str(error)) from error
            return output
        finally:
            if staged is not None:
                shutil.rmtree(staged)
