"""Copy-on-write clones with conservative full-copy fallback; never hardlinks."""

import ctypes
import os
import shutil
import sys
from pathlib import Path


def reflink(source, destination):
    """Attempt a native clone into a new path; remove partial files on failure."""
    source, destination = Path(source), Path(destination)
    try:
        if sys.platform.startswith('linux'):
            import fcntl

            with source.open('rb') as src, destination.open('xb') as dst:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # FICLONE
        elif sys.platform == 'darwin':
            library = ctypes.CDLL(None, use_errno=True)
            clone = library.clonefile
            clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            clone.restype = ctypes.c_int
            if clone(os.fsencode(source), os.fsencode(destination), 0):
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
        else:
            return False
        return True
    except (OSError, AttributeError):
        destination.unlink(missing_ok=True)
        return False


def copy_source_file(source, destination, reserve_bytes=0):
    """Clone or copy while reserving bytes for the rewritten assignments."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if reflink(source, destination):
        return True
    needed = source.stat().st_size + reserve_bytes
    free = shutil.disk_usage(destination.parent).free
    if free < needed:
        raise OSError(
            f'Insufficient disk space for full replay copy: need {needed:,} bytes, '
            f'{free:,} available.'
        )
    shutil.copyfile(source, destination)
    return False
