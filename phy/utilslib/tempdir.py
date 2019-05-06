# -*- coding: utf-8 -*-

"""Temporary directory used in unit tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import warnings as _warnings
import os as _os
import sys as _sys
from tempfile import mkdtemp

import six


#------------------------------------------------------------------------------
# Temporary directory
#------------------------------------------------------------------------------

class TemporaryDirectory(object):  # pragma: no cover
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tempdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.

    The code comes from http://stackoverflow.com/a/19299884/1595060

    """
    def __init__(self, suffix="", prefix="tmp", dir=None):
        # The tmp dir can be specified in the PHY_TMP_DIR env variable.
        dir = dir or _os.environ.get('PHY_TMP_DIR', None)
        self._closed = False
        self.name = None  # Handle mkdtemp raising an exception
        self.name = mkdtemp(suffix, prefix, dir)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        if self.name and not self._closed:
            try:
                self._rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                # Issue #10188: Emit a warning on stderr
                # if the directory could not be cleaned
                # up due to missing globals
                if "None" not in str(ex):
                    raise
                six.print_("ERROR: {!r} while cleaning up {!r}".format(ex,
                                                                       self,),
                           file=_sys.stderr)
                return
            self._closed = True
            if _warn:
                # This should be a ResourceWarning, but it is not available in
                # Python 2.x.
                self._warn("Implicitly cleaning up {!r}".format(self),
                           Warning)

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

    # XXX (ncoghlan): The following code attempts to make
    # this class tolerant of the module nulling out process
    # that happens during CPython interpreter shutdown
    # Alas, it doesn't actually manage it. See issue #10188
    _listdir = staticmethod(_os.listdir)
    _path_join = staticmethod(_os.path.join)
    _isdir = staticmethod(_os.path.isdir)
    _islink = staticmethod(_os.path.islink)
    _remove = staticmethod(_os.remove)
    _rmdir = staticmethod(_os.rmdir)
    _warn = _warnings.warn

    def _rmtree(self, path):
        # Essentially a stripped down version of shutil.rmtree.  We can't
        # use globals because they may be None'ed out at shutdown.
        for name in self._listdir(path):
            fullname = self._path_join(path, name)
            try:
                isdir = self._isdir(fullname) and not self._islink(fullname)
            except OSError:
                isdir = False
            if isdir:
                self._rmtree(fullname)
            else:
                try:
                    self._remove(fullname)
                except OSError:
                    pass
        try:
            self._rmdir(path)
        except OSError:
            pass
