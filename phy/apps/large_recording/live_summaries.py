"""Exact, bounded source scans for immutable curated unit memberships.

Instances belong to the background worker. Only completed, read-only summary
arrays cross back to the GUI; neither a session nor a model is used here.
"""

import hashlib
import json
import os
import tempfile
import zipfile
from concurrent.futures import CancelledError
from contextlib import suppress
from pathlib import Path

import numpy as np

from .curation import evaluate_predicate, validate_predicate
from .metadata import read_metadata
from .source import close_array, hash_json

VERSION = 2


def _union(parts):
    flattened = []
    for part in parts:
        flattened.extend(part['parts'] if part['kind'] == 'union' else [part])
    # Reuniting both children restores the exact parent membership, independent
    # of how the GUI allocated their transient IDs.
    siblings = {}
    for index, part in enumerate(flattened):
        if part['kind'] == 'split' and part['predicate']['nonfinite'] == 'keep':
            binding = hash_json({'parent': part['parent'], 'predicate': part['predicate']})
            previous = siblings.get(binding)
            if previous is not None and flattened[previous]['selected'] != part['selected']:
                return _union(
                    [p for i, p in enumerate(flattened) if i not in (previous, index)]
                    + [part['parent']]
                )
            siblings[binding] = index
    flattened.sort(key=hash_json)
    return flattened[0] if len(flattened) == 1 else {'kind': 'union', 'parts': flattened}


def membership_expressions(unit_ids, events):
    """Resolve journal memberships without retaining labels or allocated IDs.

    The caller supplies an already validated semantic journal. Expressions are
    JSON values rooted exclusively in the preparation's original unit IDs.
    """
    active = {int(unit): {'kind': 'source', 'unit': int(unit)} for unit in unit_ids}
    for event in events:
        operation = event['operation']
        if operation == 'merge':
            active[event['target']] = _union([active.pop(unit) for unit in event['parents']])
        elif operation == 'threshold_split':
            parent = active.pop(event['parent'])
            predicate = validate_predicate(event['predicate'])
            for selected, child in enumerate(event['children']):
                active[child] = {
                    'kind': 'split',
                    'parent': parent,
                    'predicate': predicate,
                    'selected': bool(selected),
                }
        elif operation not in ('label', 'metadata_policy'):
            raise ValueError('Unsupported membership operation.')
    return active


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise CancelledError()


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _freeze(summary):
    for array in (
        summary['counts'],
        *summary['trend_sums'].values(),
        *summary['trend_counts'].values(),
    ):
        array.flags.writeable = False
    return summary


class LiveSummaryEngine:
    """Resolve prepared/cache summaries and scan missing expressions together.

    ``max_bytes`` bounds completed outputs of one request and all resident
    derived summaries. ``cache_bytes`` bounds this package's derived disk cache.
    No cache is needed to save or replay curation.
    """

    def __init__(
        self,
        package_path,
        source_dir=None,
        batch_size=1000000,
        max_bytes=67108864,
        cache_bytes=268435456,
    ):
        self.path = Path(package_path)
        self.manifest = json.loads((self.path / '_phy_curation_manifest.json').read_text())
        self.source_dir = Path(source_dir or self.manifest['source']['path'])
        if self.source_dir.name == 'params.py':
            self.source_dir = self.source_dir.parent
        self.batch_size = int(batch_size)
        self.max_bytes, self.cache_bytes = int(max_bytes), int(cache_bytes)
        if self.batch_size <= 0 or self.max_bytes <= 0 or self.cache_bytes < 0:
            raise ValueError('Live summary budgets must be positive.')
        self.files = self.manifest.get('summaries', self.manifest)
        self.fields = sorted(key[:-9] for key in self.files if key.endswith('_sum_file'))
        self.edges = np.load(self.path / self.files['edges_file'], allow_pickle=False)
        self.exposure = np.load(self.path / self.files['exposure_file'], allow_pickle=False)
        if (
            self.edges.dtype != np.int64
            or self.edges.ndim != 1
            or len(self.edges) < 2
            or self.edges[0] != 0
            or self.edges[-1] != self.manifest['num_samples']
            or np.any(np.diff(self.edges) <= 0)
            or self.exposure.shape != (len(self.edges) - 1,)
        ):
            raise ValueError('Invalid native summary grid.')
        self.bins = len(self.edges) - 1
        self.summary_bytes = self.bins * 8 * (1 + 2 * len(self.fields))
        self.binding = {
            'version': VERSION,
            'source': self.manifest['source']['identity'],
            'revision': self.manifest['source']['revision'],
            'edges': self.edges.tolist(),
            'fields': self.fields,
        }
        self.binding_digest = hash_json(self.binding)
        self.cache_dir = self.path / '.phy' / 'live-summary-cache'
        self.arrays, self.handles, self.signatures = {}, {}, {}
        self.path_signatures = {}
        self.metadata_signature = None
        self.resident = {}

    def key(self, expression):
        return hash_json({'binding_sha256': self.binding_digest, 'expression': expression})

    def close(self):
        for array in self.arrays.values():
            close_array(array)
        for handle in self.handles.values():
            handle.close()
        self.arrays.clear()
        self.handles.clear()
        self.signatures.clear()
        self.path_signatures.clear()
        self.resident.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _empty(self):
        return {
            'counts': np.zeros(self.bins, dtype=np.int64),
            'trend_sums': {f: np.zeros(self.bins, dtype=np.float64) for f in self.fields},
            'trend_counts': {f: np.zeros(self.bins, dtype=np.int64) for f in self.fields},
        }

    def _prepared(self, expression):
        if expression['kind'] != 'source':
            return None
        row = self.manifest['unit_ids'].index(expression['unit'])
        result = self._empty()
        names = [('counts', None, self.files['counts_file'])]
        for field in self.fields:
            names.extend(
                [
                    ('trend_sums', field, self.files[f'{field}_sum_file']),
                    ('trend_counts', field, self.files[f'{field}_count_file']),
                ]
            )
        for kind, field, filename in names:
            array = np.load(self.path / filename, mmap_mode='r', allow_pickle=False)
            try:
                target = result[kind] if field is None else result[kind][field]
                if array.shape != (len(self.manifest['unit_ids']), self.bins):
                    raise ValueError('Invalid prepared summary shape.')
                target[:] = array[row]
            finally:
                close_array(array)
        return _freeze(result)

    def _flat(self, summary):
        return {
            'counts': summary['counts'],
            **{f'{f}_sum': summary['trend_sums'][f] for f in self.fields},
            **{f'{f}_count': summary['trend_counts'][f] for f in self.fields},
        }

    def _cached(self, key):
        path = self.cache_dir / f'{key}.npz'
        try:
            if path.stat().st_size > self.summary_bytes + 16384:
                return None
            expected = {'counts': np.dtype('int64')}
            for field in self.fields:
                expected[f'{field}_sum'] = np.dtype('float64')
                expected[f'{field}_count'] = np.dtype('int64')
            with path.open('rb') as stream:
                with zipfile.ZipFile(stream) as archive:
                    entries = archive.infolist()
                    names = {f'{name}.npy' for name in expected} | {'metadata.npy'}
                    if (
                        len(entries) != len(names)
                        or {info.filename for info in entries} != names
                        or sum(info.file_size for info in entries) > self.summary_bytes + 16384
                    ):
                        return None
                    for info in entries:
                        with archive.open(info) as member:
                            version = np.lib.format.read_magic(member)
                            if version == (1, 0):
                                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                            elif version == (2, 0):
                                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                            else:
                                return None
                            if info.filename == 'metadata.npy':
                                if shape != () or dtype.kind != 'U' or dtype.itemsize > 16384:
                                    return None
                                size = dtype.itemsize
                            else:
                                name = info.filename[:-4]
                                if shape != (self.bins,) or dtype != expected[name]:
                                    return None
                                size = self.bins * dtype.itemsize
                            if member.tell() + size != info.file_size:
                                return None
                stream.seek(0)
                with np.load(stream, allow_pickle=False) as record:
                    metadata = json.loads(str(record['metadata']))
                    if not isinstance(metadata, dict):
                        return None
                    result = self._empty()
                    arrays = self._flat(result)
                    if (
                        metadata.get('version') != VERSION
                        or metadata.get('key') != key
                        or set(record.files) != set(arrays) | {'metadata'}
                    ):
                        return None
                    for name, target in arrays.items():
                        value = record[name]
                        if (
                            value.shape != target.shape
                            or value.dtype != target.dtype
                            or hashlib.sha256(value.tobytes()).hexdigest()
                            != metadata['hashes'][name]
                        ):
                            return None
                        if (name == 'counts' or name.endswith('_count')) and np.any(value < 0):
                            return None
                        if not np.isfinite(value).all():
                            return None
                        target[:] = value
                    if any(
                        np.any(result['trend_counts'][f] > result['counts']) for f in self.fields
                    ):
                        return None
                    return _freeze(result)
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            return None

    def _publish(self, key, summary):
        if self.summary_bytes + 16384 > self.cache_bytes:
            return
        temporary = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            arrays = self._flat(summary)
            metadata = {
                'version': VERSION,
                'key': key,
                'hashes': {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in arrays.items()
                },
            }
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir, suffix='.tmp', delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez(stream, metadata=json.dumps(metadata), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cache_dir / f'{key}.npz')
            records = sorted(self.cache_dir.glob('*.npz'), key=lambda p: p.stat().st_mtime_ns)
            total = sum(p.stat().st_size for p in records)
            for record in records:
                if total <= self.cache_bytes:
                    break
                size = record.stat().st_size
                record.unlink()
                total -= size
        except OSError:
            pass  # Derived cache availability must not affect curation.
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def _metadata_state(self):
        names = [
            p
            for p in self.source_dir.iterdir()
            if p.name.startswith('cluster_')
            and p.stem != 'cluster_info'
            and p.suffix in ('.csv', '.tsv')
        ]
        names.extend(p for p in self.source_dir.glob('_phy_cluster_metadata*') if p.is_file())
        return {p.name: _signature(p.stat()) for p in names}

    def _unchanged(self):
        for name, handle in self.handles.items():
            expected = self.signatures[name]
            if (
                _signature(os.fstat(handle.fileno())) != expected
                or _signature((self.source_dir / name).stat()) != self.path_signatures[name]
            ):
                raise ValueError(f'Source changed during live summary: {name}.')
        if (
            self.metadata_signature is not None
            and self._metadata_state() != self.metadata_signature
        ):
            raise ValueError('Source metadata changed during live summary.')

    def _open(self, names, cancel, progress=None):
        try:
            self._unchanged()
            metadata_before = self._metadata_state()
            new_names = names - self.arrays.keys()
            total_bytes = sum((self.source_dir / n).stat().st_size for n in new_names)
            hashed_bytes = 0
            for name in sorted(new_names):
                _check_cancel(cancel)
                path = self.source_dir / name
                self.path_signatures[name] = _signature(path.stat())
                handle = path.open('rb')
                self.handles[name] = handle
                self.signatures[name] = _signature(os.fstat(handle.fileno()))
                # Windows 3.12 path stat reports creation time as ctime, while
                # fstat can report metadata change time. Bind both snapshots
                # using identity/size/mtime, and retain each provider's ctime
                # for its own subsequent stability checks.
                if self.path_signatures[name][:-1] != self.signatures[name][:-1]:
                    raise ValueError(f'Source changed during live summary: {name}.')
                digest = hashlib.sha256()
                while block := handle.read(1024 * 1024):
                    _check_cancel(cancel)
                    digest.update(block)
                    hashed_bytes += len(block)
                    if progress:
                        progress('validating source', hashed_bytes, total_bytes)
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                self.arrays[name] = array
                inventory = self.manifest['source_array_inventory'][name]
                if (
                    list(array.shape) != inventory['shape']
                    or str(array.dtype) != inventory['dtype']
                    or array.nbytes != inventory['bytes']
                    or len(array) != self.manifest['num_spikes']
                    or not np.issubdtype(array.dtype, np.number)
                ):
                    raise ValueError(f'Source shape/dtype inventory mismatch: {name}.')
                if name != 'spike_positions.npy' and not (
                    array.ndim == 1 or array.shape[1:] == (1,)
                ):
                    raise ValueError(f'Source must contain scalar rows: {name}.')
                if name != 'spike_clusters.npy':
                    expected = self.manifest['source']['identity']['arrays'][name]
                    if digest.hexdigest() != expected:
                        raise ValueError(f'Source hash mismatch: {name}.')
                else:
                    revision = hash_json(
                        {
                            'assignments': digest.hexdigest(),
                            'metadata': read_metadata(self.source_dir)[0],
                        }
                    )
                    if revision != self.manifest['source']['revision']:
                        raise ValueError('Source assignment revision mismatch.')
                self._unchanged()
            if metadata_before != self._metadata_state():
                raise ValueError('Source metadata changed during validation.')
            self.metadata_signature = metadata_before
            self._unchanged()
        except BaseException:
            self.close()
            raise

    def _required(self, expression):
        kind = expression['kind']
        if kind == 'source':
            return set()
        if kind == 'union':
            return set().union(*(self._required(p) for p in expression['parts']))
        if kind == 'split':
            return {expression['predicate']['attribute']} | self._required(expression['parent'])
        raise ValueError('Unsupported membership expression.')

    def _mask(self, expression, batch, memo=None):
        # Parents and sibling predicates are shared within a bounded batch.
        memo = {} if memo is None else memo
        key = hash_json(expression)
        if key in memo:
            return memo[key]
        kind = expression['kind']
        if kind == 'source':
            mask = batch['spike_clusters.npy'] == expression['unit']
        elif kind == 'union':
            mask = np.zeros(len(batch['spike_times.npy']), dtype=bool)
            for part in expression['parts']:
                mask |= self._mask(part, batch, memo)
        else:
            parent = self._mask(expression['parent'], batch, memo)
            predicate = expression['predicate']
            predicate_key = hash_json({'parent': expression['parent'], 'predicate': predicate})
            if predicate_key not in memo:
                memo[predicate_key] = evaluate_predicate(
                    batch[predicate['attribute']][parent], predicate
                )
            selected = memo[predicate_key]
            mask = parent.copy()
            mask[parent] = selected if expression['selected'] else ~selected
        memo[key] = mask
        return mask

    def _accumulate(self, summary, columns, selected, scalars):
        summary['counts'] += np.bincount(columns[selected], minlength=self.bins)
        for field, values in scalars.items():
            finite = selected & np.isfinite(values)
            summary['trend_sums'][field] += np.bincount(
                columns[finite], weights=values[finite].astype(np.float64), minlength=self.bins
            )
            summary['trend_counts'][field] += np.bincount(columns[finite], minlength=self.bins)

    def _chunk_size(self, expressions, row_bytes=64):
        nodes = set()

        def visit(e):
            key = hash_json(e)
            if key in nodes:
                return
            nodes.add(key)
            if e['kind'] == 'split':
                visit(e['parent'])
            elif e['kind'] == 'union':
                for part in e['parts']:
                    visit(part)

        for expression in expressions.values():
            visit(expression)
        return min(self.batch_size, max(1, self.max_bytes // (row_bytes + 2 * len(nodes))))

    def _indexed(self, expressions, cancel, progress):
        from .summary_store import StoreIntegrityError, SummaryStore

        # Template predicates are supported by the source fallback. They must
        # never be evaluated from a lossy or absent indexed scalar.
        if any(self._required(e) - {'amplitudes.npy'} for e in expressions.values()):
            return None

        def ancestors(expression):
            if expression['kind'] == 'source':
                return {expression['unit']}
            if expression['kind'] == 'split':
                return ancestors(expression['parent'])
            return set().union(*(ancestors(p) for p in expression['parts']))

        ancestry = {unit: ancestors(e) for unit, e in expressions.items()}
        originals = sorted(set().union(*ancestry.values()))
        outputs = {unit: self._empty() for unit in expressions}
        try:
            store = SummaryStore(self)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            return None
        with store:
            total = sum(store.index['units'][str(u)]['count'] for u in originals)
            done = 0
            # Bound all masks, input records and bin-index scratch space, even
            # for long journals. No recording-sized expression intermediates.
            chunk_size = self._chunk_size(expressions, store.dtype.itemsize + 64)
            for original in originals:
                targets = {u: e for u, e in expressions.items() if original in ancestry[u]}
                for data in store.batches(original, chunk_size, cancel):
                    columns = data['bin'].astype(np.intp)
                    if np.any(columns >= self.bins):
                        raise StoreIntegrityError('Summary store bin outside native grid.')
                    batch = {
                        'spike_times.npy': columns,
                        'spike_clusters.npy': np.full(len(data), original, dtype=np.int64),
                    }
                    if 'amplitude' in self.fields:
                        batch['amplitudes.npy'] = data['amplitude']
                    memo = {}
                    scalars = {field: data[field] for field in self.fields}
                    for unit, expression in targets.items():
                        _check_cancel(cancel)
                        selected = self._mask(expression, batch, memo)
                        self._accumulate(outputs[unit], columns, selected, scalars)
                    done += len(data)
                    if progress:
                        progress('reading prepared unit summaries', done, total)
            _check_cancel(cancel)
        return outputs

    def remember(self, expressions, results):
        """Remember immutable completed results within the resident byte budget."""
        for unit, value in results.items():
            if unit not in expressions:
                continue
            key = self.key(expressions[unit])
            if key not in self.resident:
                while (
                    self.resident
                    and (len(self.resident) + 1) * self.summary_bytes > self.max_bytes
                ):
                    self.resident.pop(next(iter(self.resident)))
            if self.summary_bytes <= self.max_bytes:
                self.resident[key] = _freeze(value)

    def cached(self, expressions, cancel=None):
        """Resolve available prepared, cached and additive results without source IO."""
        if len(expressions) * self.summary_bytes > self.max_bytes:
            raise ValueError('Live summary output budget exceeded.')

        def resolve_one(expression):
            _check_cancel(cancel)
            key = self.key(expression)
            if key in self.resident:
                return self.resident[key]
            value = self._prepared(expression) or self._cached(key)
            if value is None and expression['kind'] == 'union':
                value = self._empty()
                for expression_part in expression['parts']:
                    part = resolve_one(expression_part)
                    if part is None:
                        return None
                    for name, target in self._flat(value).items():
                        target += self._flat(part)[name]
                value = _freeze(value)
            return value

        results = {}
        for unit, expression in expressions.items():
            value = resolve_one(expression)
            if value is not None:
                results[unit] = value
        return results

    def resolve(self, expressions, cancel=None, progress=None):
        """Return complete summaries, scanning every unresolved membership once."""
        results = self.cached(expressions, cancel)
        missing = {
            unit: expression for unit, expression in expressions.items() if unit not in results
        }
        if not missing:
            return results
        from .summary_store import StoreIntegrityError

        try:
            indexed = self._indexed(missing, cancel, progress)
        except (OSError, StoreIntegrityError):
            # Missing, incompatible or damaged acceleration is disposable.
            # The original source retains its full validation guarantees.
            indexed = None
        if indexed is not None:
            for unit, summary in indexed.items():
                results[unit] = _freeze(summary)
                self._publish(self.key(missing[unit]), summary)
            return results
        names = {'spike_times.npy', 'spike_clusters.npy'}
        for expression in missing.values():
            names |= self._required(expression)
        if 'amplitude' in self.fields:
            names.add('amplitudes.npy')
        depth_name = 'spike_depths.npy'
        if 'depth' in self.fields:
            if depth_name not in self.manifest['source_array_inventory']:
                depth_name = 'spike_positions.npy'
            names.add(depth_name)
        self._open(names, cancel, progress=progress)
        outputs = {unit: self._empty() for unit in missing}
        try:
            chunk_size = self._chunk_size(missing, 128)
            for start in range(0, self.manifest['num_spikes'], chunk_size):
                _check_cancel(cancel)
                stop = start + chunk_size
                batch = {
                    name: array[start:stop].reshape(-1)
                    for name, array in self.arrays.items()
                    if name != 'spike_positions.npy'
                }
                times = batch['spike_times.npy']
                if (
                    not np.issubdtype(times.dtype, np.integer)
                    or np.any(times < 0)
                    or np.any(times >= self.manifest['num_samples'])
                ):
                    raise ValueError('Source spike times outside recording bounds.')
                columns = np.searchsorted(self.edges, times, side='right') - 1
                scalars = {}
                if 'amplitude' in self.fields:
                    scalars['amplitude'] = batch['amplitudes.npy']
                if 'depth' in self.fields:
                    if depth_name == 'spike_positions.npy':
                        axis = self.manifest['source']['identity']['acquisition']['depth_axis']
                        scalars['depth'] = self.arrays[depth_name][start:stop, axis]
                    else:
                        scalars['depth'] = batch[depth_name]
                memo = {}
                for unit, expression in missing.items():
                    _check_cancel(cancel)
                    selected = self._mask(expression, batch, memo)
                    self._accumulate(outputs[unit], columns, selected, scalars)
                if progress:
                    progress(
                        'scanning full source',
                        min(stop, self.manifest['num_spikes']),
                        self.manifest['num_spikes'],
                    )
            _check_cancel(cancel)
            self._unchanged()
            for unit, summary in outputs.items():
                results[unit] = _freeze(summary)
                self._publish(self.key(missing[unit]), summary)
            return results
        except BaseException:
            self.close()
            raise
