"""Memory bounds and exact row mapping for streaming accelerators."""

import numpy as np
import pytest

from phy.apps.large_recording.streaming import UnitRows, bounded_map


@pytest.mark.parametrize('units', [[0, 2, 4], [0, 2, 2147483647]])
def test_unit_rows_compact_and_sparse(units):
    lookup = UnitRows(np.array(units, dtype=np.int64))
    assert (lookup.table is None) == (units[-1] == 2147483647)
    np.testing.assert_array_equal(lookup(np.array(units[::-1])), [2, 1, 0])
    for values in ([-1], [1], [2147483648]):
        with pytest.raises(ValueError, match='catalog mismatch'):
            lookup(np.array(values))
    assert lookup(np.array([], dtype=np.int64)).size == 0


@pytest.mark.parametrize('units', [[2, 1], [1, 1], [-1, 2], [1.5]])
def test_unit_rows_reject_invalid_catalog(units):
    with pytest.raises(ValueError, match='Unit catalog'):
        UnitRows(units)


def test_bounded_ordered_map():
    assert list(bounded_map(lambda x: x * 2, range(10), workers=3)) == list(range(0, 20, 2))
    for workers in (0, 33, 1.5, True):
        with pytest.raises(ValueError, match='workers'):
            list(bounded_map(str, [], workers))


def test_parallel_hashes_equal_and_cancel(tmp_path):
    from concurrent.futures import CancelledError

    from phy.apps.large_recording.source import hash_file, hash_files

    paths = {}
    for i in range(5):
        path = tmp_path / str(i)
        path.write_bytes(bytes([i]) * (1024 * 1024 + i))
        paths[str(i)] = path
    serial = {name: hash_file(path) for name, path in paths.items()}
    progress = []
    assert hash_files(paths, workers=3, progress=lambda *args: progress.append(args)) == serial
    assert progress[-1][1:] == (sum(p.stat().st_size for p in paths.values()),) * 2
    with pytest.raises(CancelledError):
        hash_files(paths, workers=3, cancel=lambda: True)


def test_copy_fallback_space_and_source_immutability(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from phy.apps.large_recording import file_copy

    source = tmp_path / 'source'
    source.write_bytes(b'original source')
    destination = tmp_path / 'destination'
    monkeypatch.setattr(file_copy, 'reflink', lambda *_: False)
    monkeypatch.setattr(file_copy.shutil, 'disk_usage', lambda *_: SimpleNamespace(free=20))
    with pytest.raises(OSError, match='Insufficient disk space'):
        file_copy.copy_source_file(source, destination, reserve_bytes=10)
    assert not destination.exists()
    monkeypatch.setattr(file_copy.shutil, 'disk_usage', lambda *_: SimpleNamespace(free=1000))
    assert file_copy.copy_source_file(source, destination, reserve_bytes=10) is False
    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b'changed')
    assert source.read_bytes() == b'original source'


def test_native_clone_or_copy_is_independent(tmp_path):
    from phy.apps.large_recording.file_copy import copy_source_file

    source, destination = tmp_path / 'source', tmp_path / 'clone'
    source.write_bytes(b'original' * 1024)
    copy_source_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b'changed')
    assert source.read_bytes() == b'original' * 1024


def test_reflink_preserves_existing_destination(tmp_path):
    from phy.apps.large_recording.file_copy import reflink

    source, destination = tmp_path / 'source', tmp_path / 'existing'
    source.write_bytes(b'original source')
    destination.write_bytes(b'existing destination')
    with pytest.raises(FileExistsError):
        reflink(source, destination)
    assert source.read_bytes() == b'original source'
    assert destination.read_bytes() == b'existing destination'


def test_hash_rejects_concurrent_file_mutation(tmp_path):
    from phy.apps.large_recording.source import hash_file

    path = tmp_path / 'mutating'
    path.write_bytes(b'x' * 1024)
    changed = False

    def mutate(size):
        nonlocal changed
        if not changed:
            changed = True
            with path.open('ab') as stream:
                stream.write(b'changed')

    with pytest.raises(ValueError, match='changed during hashing'):
        hash_file(path, batch_bytes=512, progress=mutate)


def test_reflink_propagates_io_errors_and_cleans_owned_file(tmp_path, monkeypatch):
    import errno
    import sys

    if not sys.platform.startswith('linux'):
        pytest.skip('Linux ioctl-specific failure injection.')
    import fcntl

    from phy.apps.large_recording.file_copy import reflink

    source, destination = tmp_path / 'source', tmp_path / 'new'
    source.write_bytes(b'original')

    def fail(*args):
        raise OSError(errno.ENOSPC, 'injected no space')

    monkeypatch.setattr(fcntl, 'ioctl', fail)
    with pytest.raises(OSError, match='no space'):
        reflink(source, destination)
    assert not destination.exists()
    assert source.read_bytes() == b'original'
