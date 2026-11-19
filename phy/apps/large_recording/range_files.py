"""Bounded, selectively verified pages anchored by a published Merkle root."""

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .summary_store import StoreIntegrityError

PAGE_BYTES = 4096
_DIGEST_BYTES = 32
_EMPTY = hashlib.sha256(b'P').digest()


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def tree_leaves(size):
    pages = max(1, (int(size) + PAGE_BYTES - 1) // PAGE_BYTES)
    return 1 << (pages - 1).bit_length()


def authenticate_file(path, cancel=None, progress=None):
    """Build a packed hash tree without a file-sized mapping or heap allocation."""
    from .live_summaries import _check_cancel

    path = Path(path)
    size = path.stat().st_size
    leaves = tree_leaves(size)
    hashes = path.with_suffix(path.suffix + '.hashes')
    with path.open('rb') as source, hashes.open('w+b') as target:
        target.truncate(2 * leaves * _DIGEST_BYTES)
        target.seek(leaves * _DIGEST_BYTES)
        done = 0
        for index in range(leaves):
            if index % 256 == 0:
                _check_cancel(cancel)
            raw = source.read(PAGE_BYTES) if done < size else b''
            done += len(raw)
            target.write(hashlib.sha256(b'P' + raw).digest() if raw else _EMPTY)
            if progress and index % 4096 == 0 and done < size:
                progress('authenticating range index', min(done, size), size)
        target.flush()
        if progress:
            progress('finalizing range authentication', 0, 0)
        width = leaves // 2
        while width:
            for start in range(width, 2 * width, 8192):
                _check_cancel(cancel)
                stop = min(start + 8192, 2 * width)
                target.seek(2 * start * _DIGEST_BYTES)
                children = target.read((stop - start) * 2 * _DIGEST_BYTES)
                parents = b''.join(
                    hashlib.sha256(b'N' + children[i : i + 64]).digest()
                    for i in range(0, len(children), 64)
                )
                target.seek(start * _DIGEST_BYTES)
                target.write(parents)
            target.flush()
            width //= 2
        target.seek(_DIGEST_BYTES)
        root = target.read(_DIGEST_BYTES).hex()
        target.flush()
        os.fsync(target.fileno())
    if progress:
        progress('authenticating range index', size, size)
    return {'bytes': size, 'root': root}


class PageCache:
    """One exact byte budget shared by all files of a range reader."""

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self.size = 0
        self.pages = OrderedDict()

    def get(self, key):
        value = self.pages.get(key)
        if value is not None:
            self.pages.move_to_end(key)
        return value

    def put(self, key, value):
        if len(value) > self.max_bytes:
            return
        if key in self.pages:
            self.size -= len(self.pages.pop(key))
        while self.pages and self.size + len(value) > self.max_bytes:
            _, previous = self.pages.popitem(last=False)
            self.size -= len(previous)
        self.pages[key] = value
        self.size += len(value)

    def clear(self):
        self.pages.clear()
        self.size = 0


class VerifiedFile:
    """Verify only consumed pages; cache immutable bytes and trusted tree nodes."""

    def __init__(self, path, descriptor, cache, cancel=None):
        self.path = Path(path)
        self.size = int(descriptor['bytes'])
        self.root = bytes.fromhex(descriptor['root'])
        if self.size < 0 or len(self.root) != _DIGEST_BYTES:
            raise StoreIntegrityError('Invalid range file descriptor.')
        self.leaves = tree_leaves(self.size)
        self.cache, self.cancel = cache, cancel
        self.handles = []
        self.nodes = OrderedDict({1: self.root})
        self.bytes_read = self.pages_verified = 0
        self.hash_bytes_read = 0
        try:
            for p, expected in (
                (self.path, self.size),
                (self.path.with_suffix(self.path.suffix + '.hashes'), 2 * self.leaves * 32),
            ):
                before = _signature(p.stat())
                handle = p.open('rb')
                self.handles.append(handle)
                after = _signature(os.fstat(handle.fileno()))
                if before[:-1] != after[:-1] or after[2] != expected:
                    raise StoreIntegrityError('Range file size or identity changed.')
                if len(self.handles) == 1:
                    self.path_signature, self.fd_signature = before, after
                else:
                    self.hash_path_signature, self.hash_fd_signature = before, after
            self.data, self.hashes = self.handles
            if self._node(1) != self.root:
                raise StoreIntegrityError('Range root checksum mismatch.')
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles.clear()

    def unchanged(self):
        if (
            _signature(os.fstat(self.data.fileno())) != self.fd_signature
            or _signature(self.path.stat()) != self.path_signature
            or _signature(os.fstat(self.hashes.fileno())) != self.hash_fd_signature
            or _signature(self.path.with_suffix(self.path.suffix + '.hashes').stat())
            != self.hash_path_signature
        ):
            raise StoreIntegrityError('Range index changed during computation.')

    def _node(self, index):
        self.hashes.seek(index * 32)
        raw = self.hashes.read(32)
        self.hash_bytes_read += len(raw)
        if len(raw) != 32:
            raise StoreIntegrityError('Truncated range checksum tree.')
        return raw

    def _trust(self, index, digest):
        self.nodes[index] = digest
        self.nodes.move_to_end(index)
        while len(self.nodes) > 16384:
            self.nodes.popitem(last=False)

    def _page(self, index):
        from .live_summaries import _check_cancel

        _check_cancel(self.cancel)
        key = (self.root, self.size, index)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.data.seek(index * PAGE_BYTES)
        raw = self.data.read(min(PAGE_BYTES, self.size - index * PAGE_BYTES))
        if not raw or len(raw) != min(PAGE_BYTES, self.size - index * PAGE_BYTES):
            raise StoreIntegrityError('Truncated range data page.')
        self.bytes_read += len(raw)
        node = self.leaves + index
        digest = hashlib.sha256(b'P' + raw).digest()
        pending = [(node, digest)]
        while node != 1 and node not in self.nodes:
            sibling = node ^ 1
            other = self._node(sibling)
            pending.append((sibling, other))
            digest = hashlib.sha256(
                b'N' + (other + digest if node & 1 else digest + other)
            ).digest()
            node //= 2
            pending.append((node, digest))
        expected = self.root if node == 1 else self.nodes[node]
        if digest != expected:
            raise StoreIntegrityError('Range page checksum mismatch.')
        for index_, value in pending:
            self._trust(index_, value)
        self.cache.put(key, raw)
        self.pages_verified += 1
        return raw

    def read(self, offset, size):
        offset, size = int(offset), int(size)
        if offset < 0 or size < 0 or offset + size > self.size:
            raise StoreIntegrityError('Range read exceeds the published file.')
        if not size:
            return b''
        chunks = []
        while size:
            page, within = divmod(offset, PAGE_BYTES)
            raw = self._page(page)
            count = min(size, len(raw) - within)
            chunks.append(raw[within : within + count])
            offset += count
            size -= count
        return b''.join(chunks)

    def gather(self, offsets, dtype):
        """Read aligned scalar/record locations, grouping requests by verified page."""
        dtype = np.dtype(dtype)
        offsets = np.asarray(offsets, dtype=np.int64)
        if np.any(offsets < 0) or np.any(offsets > self.size - dtype.itemsize):
            raise StoreIntegrityError('Range gather exceeds the published file.')
        output = np.empty(len(offsets), dtype=dtype)
        order = np.argsort(offsets // PAGE_BYTES, kind='stable')
        page_ids = offsets[order] // PAGE_BYTES
        boundaries = np.r_[0, np.flatnonzero(np.diff(page_ids)) + 1, len(order)]
        for first, stop in zip(boundaries[:-1], boundaries[1:]):
            if first == stop:
                continue
            selected = order[first:stop]
            page = page_ids[first]
            positions = offsets[selected] % PAGE_BYTES
            raw = self._page(int(page))
            if len(selected) == 1 and int(positions[0]) + dtype.itemsize <= len(raw):
                output[selected[0]] = np.frombuffer(
                    raw, dtype=dtype, count=1, offset=int(positions[0])
                )[0]
            elif np.any(positions + dtype.itemsize > len(raw)):
                for item in selected:
                    output[item] = np.frombuffer(
                        self.read(offsets[item], dtype.itemsize), dtype=dtype
                    )[0]
            else:
                # Records and scalar keys are aligned by the builder.
                if np.any(positions % dtype.itemsize):
                    for item in selected:
                        output[item] = np.frombuffer(
                            self.read(offsets[item], dtype.itemsize), dtype=dtype
                        )[0]
                else:
                    usable = len(raw) // dtype.itemsize * dtype.itemsize
                    output[selected] = np.frombuffer(raw[:usable], dtype=dtype)[
                        positions // dtype.itemsize
                    ]
        return output
