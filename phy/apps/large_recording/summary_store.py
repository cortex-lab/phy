"""Optional, independently budgeted unit-contiguous summary snapshot storage."""

import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import numpy as np

from .curation import _commit_lock, _fsync_directory, _safe_file
from .source import hash_json, publish_directory

VERSION = 1
REFERENCE = 'live-summary-store.json'


def binding(engine):
    """Bind a store to the same exact source snapshot and native grid as summaries."""
    return {'version': VERSION, 'summary_binding': engine.binding}


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, sort_keys=True).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _bin_dtype(bins):
    for dtype in ('uint8', 'uint16', 'uint32', 'uint64'):
        if bins - 1 <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError('Too many native summary bins.')


@dataclass(frozen=True)
class StoreEstimate:
    payload_bytes: int
    estimated_bytes: int
    source_spikes: int
    record_bytes: int
    source_path: str


class SummaryStorePublishedError(OSError):
    """A completed artifact remains available after finalization/attachment failed."""

    def __init__(self, path, message):
        self.path = Path(path)
        super().__init__(message)


def _layout(engine):
    inventory = engine.manifest['source_array_inventory']
    names = {'spike_times.npy', 'spike_clusters.npy'}
    specs = [('bin', _bin_dtype(engine.bins))]
    scalar_names = {}
    for field in engine.fields:
        name = 'amplitudes.npy' if field == 'amplitude' else 'spike_depths.npy'
        if name not in inventory and field == 'depth':
            name = 'spike_positions.npy'
        names.add(name)
        scalar_names[field] = name
        specs.append((field, np.dtype(inventory[name]['dtype'])))
    return np.dtype(specs), names, scalar_names


def estimate_summary_store(package_path, source_dir=None):
    """Read package metadata only; use exactly the builder's disk calculation."""
    from .live_summaries import LiveSummaryEngine

    with LiveSummaryEngine(package_path, source_dir=source_dir) as engine:
        dtype, _, _ = _layout(engine)
        count = int(engine.manifest['num_spikes'])
        payload = count * dtype.itemsize
        return StoreEstimate(
            payload,
            payload + 65536 + 512 * len(engine.manifest['unit_ids']),
            count,
            dtype.itemsize,
            str(engine.source_dir),
        )


def build_summary_store(
    package_path,
    output_dir,
    source_dir=None,
    max_bytes=68719476736,
    batch_size=1000000,
    buffer_bytes=33554432,
    cancel=None,
    progress=None,
    attach=True,
    before_publish=None,
):
    """Build an exact prepared-snapshot asset without modifying saved curation.

    The caller selects a new directory outside the bounded package. Source
    validation is mandatory. Unit ranges preserve original row order. Bounded
    batches and a globally capped write buffer avoid a recording-sized sort or
    permutation. Publication never replaces an existing store.
    """
    from .live_summaries import LiveSummaryEngine, _check_cancel

    package_path, output_dir = Path(package_path).resolve(), Path(output_dir).resolve()
    if output_dir == package_path or package_path in output_dir.parents:
        raise ValueError('Summary store must be outside the bounded package.')
    if output_dir.exists():
        raise ValueError(f'Summary store already exists: {output_dir}.')
    if min(max_bytes, batch_size, buffer_bytes) <= 0:
        raise ValueError('Summary store budgets must be positive.')
    temporary = None
    with LiveSummaryEngine(package_path, source_dir=source_dir, batch_size=batch_size) as engine:
        source_path = engine.source_dir.resolve()
        if output_dir == source_path or source_path in output_dir.parents:
            raise ValueError('Summary store must be outside the original source.')
        dtype, names, scalar_names = _layout(engine)
        count = int(engine.manifest['num_spikes'])
        payload_bytes = count * dtype.itemsize
        # Includes the small catalog and headers, independently of bounded budgets.
        estimate = payload_bytes + 65536 + 512 * len(engine.manifest['unit_ids'])
        if estimate > max_bytes:
            raise ValueError(
                f'Summary store needs approximately {estimate:,} bytes; increase max_bytes.'
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output_dir.parent).free < estimate:
            raise OSError(f'Insufficient free space for summary store: need {estimate:,} bytes.')
        engine._open(names, cancel, progress=progress)
        units = np.asarray(engine.manifest['unit_ids'], dtype=np.int64)
        counts = np.asarray(engine.manifest['counts'], dtype=np.int64)
        if np.any(counts < 0) or sum(map(int, counts)) != count:
            raise ValueError('Invalid source unit counts.')
        offsets = np.zeros(len(units) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        cursors = offsets[:-1].copy()
        pending = np.zeros(len(units), dtype=np.int64)
        digests = [hashlib.sha256() for _ in units]
        buffers, buffered = OrderedDict(), 0
        temporary = Path(
            tempfile.mkdtemp(prefix=f'.{output_dir.name}.building-', dir=output_dir.parent)
        )
        try:
            with (temporary / 'data.bin').open('w+b') as stream:
                stream.truncate(payload_bytes)

                def flush(unit):
                    nonlocal buffered
                    chunks = buffers.pop(unit)
                    stream.seek(int(cursors[unit]) * dtype.itemsize)
                    for chunk in chunks:
                        stream.write(chunk)
                        digests[unit].update(chunk)
                        cursors[unit] += len(chunk) // dtype.itemsize
                        buffered -= len(chunk)
                        pending[unit] -= len(chunk) // dtype.itemsize

                def read_rows(name, start, stop, axis=None):
                    array = engine.arrays[name]
                    width = int(np.prod(array.shape[1:]))
                    handle = engine.handles[name]
                    if axis is not None and array.flags.f_contiguous:
                        handle.seek(
                            array.offset + (axis * len(array) + start) * array.dtype.itemsize
                        )
                        raw = handle.read((stop - start) * array.dtype.itemsize)
                        if len(raw) != (stop - start) * array.dtype.itemsize:
                            raise ValueError('Source truncated during store construction.')
                        return np.frombuffer(raw, dtype=array.dtype)
                    handle.seek(array.offset + start * width * array.dtype.itemsize)
                    size = (stop - start) * width * array.dtype.itemsize
                    raw = handle.read(size)
                    if len(raw) != size:
                        raise ValueError('Source truncated during store construction.')
                    values = np.frombuffer(raw, dtype=array.dtype).reshape(
                        (stop - start, *array.shape[1:])
                    )
                    return values[:, axis] if axis is not None else values

                for start in range(0, count, batch_size):
                    _check_cancel(cancel)
                    stop = min(count, start + batch_size)
                    times = read_rows('spike_times.npy', start, stop).reshape(-1)
                    if np.any(times < 0) or np.any(times >= engine.manifest['num_samples']):
                        raise ValueError('Source spike times outside recording bounds.')
                    assignments = read_rows('spike_clusters.npy', start, stop).reshape(-1)
                    rows = np.searchsorted(units, assignments)
                    if np.any(rows >= len(units)) or np.any(units[rows] != assignments):
                        raise ValueError('Source unit catalog mismatch.')
                    order = np.argsort(rows, kind='stable')
                    sorted_rows = rows[order]
                    data = np.empty(stop - start, dtype=dtype)
                    data['bin'] = np.searchsorted(engine.edges, times, side='right') - 1
                    for field, name in scalar_names.items():
                        axis = (
                            engine.manifest['source']['identity']['acquisition']['depth_axis']
                            if name == 'spike_positions.npy'
                            else None
                        )
                        values = read_rows(name, start, stop, axis=axis)
                        data[field] = values.reshape(-1)
                    data = data[order]
                    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_rows)) + 1, len(data)]
                    for first, last in zip(boundaries[:-1], boundaries[1:]):
                        unit = int(sorted_rows[first])
                        if cursors[unit] + pending[unit] + last - first > offsets[unit + 1]:
                            raise ValueError('Source unit count mismatch.')
                        chunk = data[first:last].tobytes()
                        if len(chunk) >= buffer_bytes:
                            if unit in buffers:
                                flush(unit)
                            stream.seek(int(cursors[unit]) * dtype.itemsize)
                            stream.write(chunk)
                            digests[unit].update(chunk)
                            cursors[unit] += last - first
                            continue
                        while buffers and buffered + len(chunk) > buffer_bytes:
                            flush(next(iter(buffers)))
                        buffers.setdefault(unit, []).append(chunk)
                        buffered += len(chunk)
                        pending[unit] += last - first
                        if buffered >= buffer_bytes:
                            flush(unit)
                    if progress:
                        progress('building summary store', stop, count)
                if progress:
                    progress('finalizing summary store', 0, 0)
                while buffers:
                    _check_cancel(cancel)
                    flush(next(iter(buffers)))
                if not np.array_equal(cursors, offsets[1:]):
                    raise ValueError('Source unit counts changed.')
                stream.flush()
                os.fsync(stream.fileno())
            engine._unchanged()
            catalog = {
                'binding': binding(engine),
                'dtype': dtype.descr,
                'payload_bytes': payload_bytes,
                'units': {
                    str(int(unit)): {
                        'start': int(offsets[i]),
                        'count': int(counts[i]),
                        'sha256': digests[i].hexdigest(),
                    }
                    for i, unit in enumerate(units)
                },
            }
            _atomic_json(temporary / 'index.json', catalog)
            actual_bytes = sum(p.stat().st_size for p in temporary.iterdir())
            if actual_bytes > max_bytes:
                raise ValueError(f'Summary store exceeds max_bytes: {actual_bytes:,} bytes.')
            _check_cancel(cancel)
            # A GUI task may enter its noncancellable publication boundary here.
            if before_publish is not None:
                before_publish()
            publish_directory(temporary, output_dir)
            temporary = None
            try:
                _fsync_directory(output_dir.parent)
                if attach:
                    if progress:
                        progress('attaching summary store', 0, 0)
                    attach_summary_store(package_path, output_dir)
            except (OSError, ValueError) as error:
                raise SummaryStorePublishedError(output_dir, str(error)) from error
            return output_dir
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)


class StoreIntegrityError(ValueError):
    """Disposable acceleration data failed its integrity contract."""


class SummaryStore:
    """Read and verify only requested units from a prepared source snapshot.

    This certifies the indexed snapshot, not the current contents of original
    source files. Every consumed unit is hashed before its results can publish.
    """

    def __init__(self, engine, path=None):
        reference = (
            json.loads(summary_store_reference(engine.path))
            if path is None
            else {'path': str(path)}
        )
        self.path = Path(reference['path'])
        index_path = self.path / 'index.json'
        if index_path.stat().st_size > 16777216:
            raise StoreIntegrityError('Summary store catalog exceeds limit.')
        self.index = json.loads(index_path.read_text())
        if (
            'index_sha256' in reference and hash_json(self.index) != reference['index_sha256']
        ) or self.index['binding'] != binding(engine):
            raise StoreIntegrityError('Summary store binding mismatch.')
        self.dtype = np.dtype([tuple(s) for s in self.index['dtype']])
        if self.dtype.hasobject or self.dtype.names != ('bin', *engine.fields):
            raise StoreIntegrityError('Invalid summary store dtype.')
        expected = [('bin', _bin_dtype(engine.bins))]
        inventory = engine.manifest['source_array_inventory']
        for field in engine.fields:
            name = 'amplitudes.npy' if field == 'amplitude' else 'spike_depths.npy'
            if name not in inventory and field == 'depth':
                name = 'spike_positions.npy'
            expected.append((field, np.dtype(inventory[name]['dtype'])))
        if self.dtype != np.dtype(expected):
            raise StoreIntegrityError('Summary store scalar dtype mismatch.')
        end = 0
        for unit, count in zip(engine.manifest['unit_ids'], engine.manifest['counts']):
            item = self.index['units'][str(unit)]
            if item['start'] != end or item['count'] != count:
                raise StoreIntegrityError('Summary store unit ranges mismatch.')
            end += int(count)
        if (
            set(self.index['units']) != set(map(str, engine.manifest['unit_ids']))
            or end * self.dtype.itemsize != self.index['payload_bytes']
        ):
            raise StoreIntegrityError('Summary store payload mismatch.')
        self.stream = (self.path / 'data.bin').open('rb')
        try:
            self.signature = os.fstat(self.stream.fileno())
            self.path_signature = (self.path / 'data.bin').stat()
            if self.signature.st_size != self.index['payload_bytes']:
                raise StoreIntegrityError('Summary store payload size mismatch.')
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stream.close()

    def batches(self, unit, batch_size, cancel=None):
        from .live_summaries import _check_cancel, _signature

        item = self.index['units'][str(unit)]
        digest = hashlib.sha256()
        self.stream.seek(item['start'] * self.dtype.itemsize)
        for start in range(0, item['count'], batch_size):
            _check_cancel(cancel)
            size = min(batch_size, item['count'] - start) * self.dtype.itemsize
            raw = self.stream.read(size)
            if len(raw) != size:
                raise StoreIntegrityError('Truncated summary store.')
            digest.update(raw)
            yield np.frombuffer(raw, dtype=self.dtype)
        if digest.hexdigest() != item['sha256']:
            raise StoreIntegrityError('Summary store unit checksum mismatch.')
        if _signature(os.fstat(self.stream.fileno())) != _signature(self.signature) or _signature(
            (self.path / 'data.bin').stat()
        ) != _signature(self.path_signature):
            raise StoreIntegrityError('Summary store changed while reading.')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def _attachment_lock(path, check_cancel=None):
    if check_cancel is None:
        with _commit_lock(path):
            yield
        return
    with _safe_file(path, 'commit.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'\0')
                stream.flush()

            def acquire():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        while True:
            check_cancel()
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                sleep(0.05)
        try:
            check_cancel()
            yield
        finally:
            release()


_ANY_REFERENCE = object()


@dataclass(frozen=True)
class ReferenceStamp:
    signature: tuple


def summary_store_reference(package_path, allow_invalid=False):
    path = Path(package_path) / '.phy' / REFERENCE
    try:
        stat = path.stat()
        if stat.st_size > 65536:
            if allow_invalid:
                return ReferenceStamp(
                    (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                )
            raise ValueError('Summary store reference exceeds limit.')
        return path.read_bytes()
    except FileNotFoundError:
        return None


def attach_summary_store(
    package_path,
    store_path,
    expected_reference=_ANY_REFERENCE,
    before_attach=None,
    check_cancel=None,
):
    """Attach a snapshot under a short cross-process reference lock.

    An optional expected reference prevents overwriting a newer user choice.
    Payload integrity checks remain selective during use.
    """
    from .live_summaries import LiveSummaryEngine

    with LiveSummaryEngine(package_path) as engine, SummaryStore(engine, path=store_path) as store:
        lock_path = engine.path / '.phy' / 'summary-store-attachment'
        lock_path.mkdir(parents=True, exist_ok=True)
        with _attachment_lock(lock_path, check_cancel):
            if (
                expected_reference is not _ANY_REFERENCE
                and summary_store_reference(engine.path, allow_invalid=True) != expected_reference
            ):
                raise ValueError(
                    'The attached summary store changed while this build was running; attach the completed store manually.'
                )
            if before_attach is not None:
                before_attach()
            _atomic_json(
                engine.path / '.phy' / REFERENCE,
                {
                    'path': str(store.path.resolve()),
                    'index_sha256': hash_json(store.index),
                },
            )
    return Path(store_path)
