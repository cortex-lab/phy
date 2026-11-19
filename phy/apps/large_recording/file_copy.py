"""Copy-on-write clones with conservative full-copy fallback; never hardlinks."""

import ctypes
import errno
import os
import shutil
import sys
from pathlib import Path


def reflink(source, destination):
    """Attempt a native clone; never remove a destination owned by a caller."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    created = False
    unsupported = {errno.EXDEV, errno.ENOTTY, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}
    try:
        if sys.platform.startswith('linux'):
            import fcntl

            with source.open('rb') as src, destination.open('xb') as dst:
                created = True
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # FICLONE
        elif sys.platform == 'darwin':
            library = ctypes.CDLL(None, use_errno=True)
            clone = getattr(library, 'clonefile', None)
            if clone is None:
                return False
            clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            clone.restype = ctypes.c_int
            if clone(os.fsencode(source), os.fsencode(destination), 0):
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
        else:
            return False
        return True
    except OSError as error:
        # Only Linux's successful exclusive open establishes ownership. In
        # particular EEXIST must leave an existing destination untouched.
        if created:
            destination.unlink(missing_ok=True)
        if error.errno not in unsupported or destination.exists():
            raise
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
