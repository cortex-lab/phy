"""Sublinear amplitude interval queries over an authenticated prepared snapshot."""

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .curation import evaluate_predicate
from .range_build import FORMAT, VERSION, layout
from .range_files import PageCache, VerifiedFile
from .range_intervals import IntervalProgram, RangeUnavailableError
from .range_math import add_stats, row_stats, validate_sums
from .source import hash_json
from .summary_store import (
    _ANY_REFERENCE,
    StoreIntegrityError,
    _atomic_json,
    _attachment_lock,
    summary_store_reference,
)


class AmplitudeRangeIndex:
    """Worker-owned readers; all consumed pages are verified against pinned roots."""

    def __init__(self, engine, path=None):
        self.engine = engine
        self.files = {}
        self.cache = PageCache(engine.max_bytes)
        self.endpoint_bytes = 0
        reference = json.loads(summary_store_reference(engine.path))
        index_reference = reference.get('amplitude_index', {})
        self.path = Path(path if path is not None else index_reference['path'])
        if not self.path.is_absolute():
            self.path = Path(reference['path']) / self.path
        catalog_path = self.path / 'range.json'
        if catalog_path.stat().st_size > 2097152:
            raise StoreIntegrityError('Range catalog exceeds limit.')
        self.catalog = json.loads(catalog_path.read_text())
        self.catalog_sha256 = hash_json(self.catalog)
        if path is None and self.catalog_sha256 != index_reference.get('catalog_sha256'):
            raise StoreIntegrityError('Range catalog checksum mismatch.')
        if (
            self.catalog['format'] != FORMAT
            or self.catalog['version'] != VERSION
            or self.catalog['arithmetic_version'] != 1
            or self.catalog['binding_sha256'] != engine.binding_digest
            or self.catalog['parent_sha256'] != reference['index_sha256']
        ):
            raise StoreIntegrityError('Range index snapshot binding mismatch.')
        self.payload, self.directory_dtype, self.node = layout(engine)
        self.fields = engine.fields
        self.block_rows = self.catalog['block_rows']
        if type(self.block_rows) is not int or not 1 <= self.block_rows <= 65536:
            raise StoreIntegrityError('Invalid range block size.')
        self.ordinals = {int(unit): i for i, unit in enumerate(engine.manifest['unit_ids'])}
        self.indexed_units = set(self.catalog['indexed_units'])
        if not self.indexed_units <= self.ordinals.keys():
            raise StoreIntegrityError('Range catalog contains unknown units.')
        expected_directory = len(self.ordinals) * engine.bins * self.directory_dtype.itemsize
        if self.catalog['files']['segments']['bytes'] != expected_directory:
            raise StoreIntegrityError('Range directory size mismatch.')
        max_payload = (
            sum(
                int(n)
                for u, n in zip(engine.manifest['unit_ids'], engine.manifest['counts'])
                if u in self.indexed_units
            )
            * self.payload.itemsize
        )
        if self.catalog['files']['values']['bytes'] > max_payload:
            raise StoreIntegrityError('Range values exceed source inventory.')
        try:
            for name in ('values', 'nodes', 'segments'):
                self.files[name] = VerifiedFile(
                    self.path / f'{name}.bin', self.catalog['files'][name], self.cache
                )
        except BaseException:
            self.close()
            raise
        self.last_metrics = {}

    def close(self):
        for file in self.files.values():
            file.close()
        self.files.clear()
        self.cache.clear()
        self.endpoint_bytes = 0

    def unchanged(self):
        for file in self.files.values():
            file.unchanged()

    def _block(self, directory, block, segment, cache):
        key = (int(segment), int(block))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        count = min(self.block_rows, int(directory['finite']) - int(block) * self.block_rows)
        if count <= 0:
            raise StoreIntegrityError('Range endpoint block exceeds its segment.')
        offset = int(directory['start']) + int(block) * self.block_rows * self.payload.itemsize
        records = np.frombuffer(
            self.files['values'].read(offset, count * self.payload.itemsize), dtype=self.payload
        )
        if np.any(~np.isfinite(records['amplitude'])):
            raise StoreIntegrityError('Range finite partition contains nonfinite values.')
        while cache and self.endpoint_bytes + records.nbytes > 8388608:
            self.endpoint_bytes -= cache.popitem(last=False)[1].nbytes
        self.endpoint_bytes += records.nbytes
        cache[key] = records
        return records

    def _cuts(self, directories, segments, predicate, cache, cancel):
        from .live_summaries import _check_cancel

        count = directories['finite']
        blocks = (count + self.block_rows - 1) // self.block_rows
        lo = np.zeros(len(directories), dtype=np.int64)
        hi = blocks.copy()
        dtype = self.payload['amplitude']
        while np.any(lo < hi):
            _check_cancel(cancel)
            selected = np.flatnonzero(lo < hi)
            middle = (lo[selected] + hi[selected]) // 2
            values = self.files['nodes'].gather(
                directories['keys'][selected] + middle * dtype.itemsize, dtype
            )
            if np.any(~np.isfinite(values)):
                raise StoreIntegrityError('Range boundary keys are not finite.')
            before = evaluate_predicate(values, predicate)
            lo[selected[before]] = middle[before] + 1
            hi[selected[~before]] = middle[~before]
        candidate = lo - 1
        ranks = np.zeros(len(directories), dtype=np.int64)
        selected = np.flatnonzero(candidate >= 0)
        if len(selected):
            rows, valid = self._blocks(
                directories[selected], candidate[selected], segments[selected], cache, cancel
            )
            ranks[selected] = candidate[selected] * self.block_rows + (
                evaluate_predicate(rows['amplitude'], predicate).reshape(rows.shape) & valid
            ).sum(axis=1)
        return ranks

    def _blocks(self, directories, blocks, segments, cache, cancel):
        from .live_summaries import _check_cancel

        rows = np.zeros((len(directories), self.block_rows), dtype=self.payload)
        valid = np.zeros(rows.shape, dtype=bool)
        for i in range(len(directories)):
            _check_cancel(cancel)
            records = self._block(directories[i], blocks[i], segments[i], cache)
            rows[i, : len(records)] = records
            valid[i, : len(records)] = True
        return rows, valid

    def _tree_ranges(self, directories, first, stop, cancel):
        from .live_summaries import _check_cancel

        output = np.zeros(len(directories), dtype=self.node)
        left = first + directories['leaves']
        right = stop + directories['leaves']
        while np.any(left < right):
            _check_cancel(cancel)
            use_left = np.flatnonzero((left < right) & ((left & 1) != 0))
            if len(use_left):
                offsets = directories['tree'][use_left] + left[use_left] * self.node.itemsize
                values = self.files['nodes'].gather(offsets, self.node)
                output[use_left] = add_stats(output[use_left], values, self.fields)
                left[use_left] += 1
            use_right = np.flatnonzero((left < right) & ((right & 1) != 0))
            if len(use_right):
                right[use_right] -= 1
                offsets = directories['tree'][use_right] + right[use_right] * self.node.itemsize
                values = self.files['nodes'].gather(offsets, self.node)
                output[use_right] = add_stats(output[use_right], values, self.fields)
            left //= 2
            right //= 2
        return output

    def _ranges(self, directories, segments, lower, upper, cache, cancel):
        from .live_summaries import _check_cancel

        first = lower // self.block_rows
        stop = upper // self.block_rows
        partial_left = lower % self.block_rows != 0
        first = first + partial_left
        full_last = (upper == directories['finite']) & (upper % self.block_rows != 0)
        stop = stop + full_last
        output = self._tree_ranges(directories, np.minimum(first, stop), stop, cancel)
        positions = np.arange(self.block_rows)
        for chosen, blocks, starts, ends in [
            (
                partial_left,
                lower // self.block_rows,
                lower % self.block_rows,
                np.minimum(self.block_rows, upper - (lower // self.block_rows) * self.block_rows),
            ),
            (
                (upper % self.block_rows != 0) & ~full_last & (stop >= first),
                stop,
                np.zeros(len(stop), dtype=np.int64),
                upper % self.block_rows,
            ),
        ]:
            selected = np.flatnonzero(chosen)
            if not len(selected):
                continue
            rows, valid = self._blocks(
                directories[selected], blocks[selected], segments[selected], cache, cancel
            )
            valid &= (positions >= starts[selected, None]) & (positions < ends[selected, None])
            output[selected] = add_stats(
                output[selected], row_stats(rows, self.fields, self.node, valid), self.fields
            )
        if np.any(output['amplitude_count'] != upper - lower):
            raise StoreIntegrityError('Range aggregate rank/count mismatch.')
        return output

    def resolve(self, expressions, cancel=None, progress=None):
        from .live_summaries import _check_cancel

        program = IntervalProgram(expressions)
        if not set(program.originals) <= self.indexed_units:
            raise RangeUnavailableError('Some original units use the row fallback.')
        self.unchanged()
        for file in self.files.values():
            file.cancel = cancel
        before = {
            name: (f.bytes_read, f.hash_bytes_read, f.pages_verified)
            for name, f in self.files.items()
        }
        if len(expressions) * self.engine.bins * 24 * len(self.fields) > self.engine.max_bytes:
            raise RangeUnavailableError('Range arithmetic scratch budget exceeded.')
        outputs = {u: self.engine._empty() for u in expressions}
        absolute = {u: {f: np.zeros(self.engine.bins) for f in self.fields} for u in expressions}
        errors = {u: {f: np.zeros(self.engine.bins) for f in self.fields} for u in expressions}
        operations = {
            u: {f: np.zeros(self.engine.bins, dtype=np.int64) for f in self.fields}
            for u in expressions
        }
        chunk_bins = min(
            128,
            max(
                1,
                self.engine.max_bytes
                // (
                    max(1, len(program.cuts))
                    * (self.block_rows * (self.payload.itemsize + 40) + 256)
                ),
            ),
        )
        entry_limit = max(
            1,
            self.engine.max_bytes
            // (
                4 * self.directory_dtype.itemsize
                + 5 * self.node.itemsize
                + 160
                + self.block_rows * (self.payload.itemsize + 40)
            ),
        )
        endpoint_cache = OrderedDict()
        self.endpoint_bytes = 0
        all_units = list(expressions)
        done = 0
        total = len(program.originals) * self.engine.bins
        for original in program.originals:
            ordinal = self.ordinals[original]
            for first in range(0, self.engine.bins, chunk_bins):
                _check_cancel(cancel)
                count = min(chunk_bins, self.engine.bins - first)
                segments = ordinal * self.engine.bins + np.arange(
                    first, first + count, dtype=np.int64
                )
                directories = np.frombuffer(
                    self.files['segments'].read(
                        int(segments[0]) * self.directory_dtype.itemsize,
                        count * self.directory_dtype.itemsize,
                    ),
                    dtype=self.directory_dtype,
                )
                if np.any(directories['finite'] < 0) or np.any(
                    directories['total'] < directories['finite']
                ):
                    raise StoreIntegrityError('Range segment counts are invalid.')
                cuts = {
                    key: self._cuts(directories, segments, predicate, endpoint_cache, cancel)
                    for key, predicate in program.cuts.items()
                }
                targets, bins, which, lower, upper = [], [], [], [], []
                for i, directory in enumerate(directories):
                    states = program.evaluate(
                        original,
                        int(directory['finite']),
                        int(directory['total'] - directory['finite']),
                        {key: int(value[i]) for key, value in cuts.items()},
                        interval_limit=max(128, self.engine.max_bytes // 96),
                    )
                    for unit, (parts, nonfinite) in states.items():
                        if nonfinite:
                            outputs[unit]['counts'][first + i] += (
                                directory['total'] - directory['finite']
                            )
                            if 'depth' in self.fields:
                                for kind, sink in [
                                    ('sum', outputs[unit]['trend_sums']),
                                    ('count', outputs[unit]['trend_counts']),
                                    ('abs', absolute[unit]),
                                    ('error', errors[unit]),
                                ]:
                                    sink['depth'][first + i] += directory['nf_depth_' + kind]
                                operations[unit]['depth'][first + i] += 1
                        for lo, hi in parts:
                            targets.append(unit)
                            bins.append(first + i)
                            which.append(i)
                            lower.append(lo)
                            upper.append(hi)
                            if len(targets) > entry_limit:
                                raise RangeUnavailableError(
                                    'Range interval scratch budget exceeded.'
                                )
                if targets:
                    selected = np.asarray(which, dtype=np.int64)
                    stats = self._ranges(
                        directories[selected],
                        segments[selected],
                        np.asarray(lower, dtype=np.int64),
                        np.asarray(upper, dtype=np.int64),
                        endpoint_cache,
                        cancel,
                    )
                    targets = np.asarray(targets)
                    bins = np.asarray(bins, dtype=np.int64)
                    for unit in all_units:
                        chosen = targets == unit
                        if not np.any(chosen):
                            continue
                        indices = bins[chosen]
                        np.add.at(
                            outputs[unit]['counts'], indices, stats['amplitude_count'][chosen]
                        )
                        for field in self.fields:
                            np.add.at(
                                outputs[unit]['trend_sums'][field],
                                indices,
                                stats[field + '_sum'][chosen],
                            )
                            np.add.at(
                                outputs[unit]['trend_counts'][field],
                                indices,
                                stats[field + '_count'][chosen],
                            )
                            np.add.at(
                                absolute[unit][field], indices, stats[field + '_abs'][chosen]
                            )
                            np.add.at(
                                errors[unit][field], indices, stats[field + '_error'][chosen]
                            )
                            np.add.at(operations[unit][field], indices, 1)
                done += count
                if progress:
                    progress('querying amplitude ranges', done, total)
        _check_cancel(cancel)
        self.unchanged()
        for unit in outputs:
            validate_sums(
                outputs[unit], absolute[unit], errors[unit], operations[unit], self.fields
            )
        self.last_metrics = {
            'payload_bytes_read': sum(f.bytes_read - before[n][0] for n, f in self.files.items()),
            'authentication_bytes_read': sum(
                f.hash_bytes_read - before[n][1] for n, f in self.files.items()
            ),
            'pages_verified': sum(f.pages_verified - before[n][2] for n, f in self.files.items()),
            'resident_page_bytes': self.cache.size,
        }
        return outputs


def attach_amplitude_index(
    package_path,
    index_path,
    expected_reference=_ANY_REFERENCE,
    before_attach=None,
    check_cancel=None,
):
    """Attach only to the matching prepared row snapshot, preserving newer choices."""
    from .live_summaries import LiveSummaryEngine

    with LiveSummaryEngine(package_path) as engine:
        index = AmplitudeRangeIndex(engine, path=index_path)
        try:
            lock = engine.path / '.phy' / 'summary-store-attachment'
            lock.mkdir(parents=True, exist_ok=True)
            with _attachment_lock(lock, check_cancel):
                raw = summary_store_reference(engine.path, allow_invalid=True)
                if expected_reference is not _ANY_REFERENCE and raw != expected_reference:
                    raise ValueError(
                        'The attached store changed during amplitude-index preparation.'
                    )
                reference = json.loads(raw)
                if reference['index_sha256'] != index.catalog['parent_sha256']:
                    raise ValueError('The amplitude index needs its original row store.')
                if before_attach:
                    before_attach()
                path = Path(index_path).resolve()
                try:
                    path_text = str(path.relative_to(Path(reference['path']).resolve()))
                except ValueError:
                    path_text = str(path)
                reference['amplitude_index'] = {
                    'path': path_text,
                    'catalog_sha256': index.catalog_sha256,
                }
                _atomic_json(engine.path / '.phy' / 'live-summary-store.json', reference)
        finally:
            index.close()
    return Path(index_path)
