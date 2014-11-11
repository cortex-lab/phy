# -*- coding: utf-8 -*-

"""HDF5 input and output."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
try:
    import h5py
except ImportError as exception:
    # TODO: logging.
    raise exception

from ..ext import six


#------------------------------------------------------------------------------
# HDF5 routines
#------------------------------------------------------------------------------

class File(object):
    def __init__(self, filename, mode=None):
        if mode is None:
            mode = 'r'
        self.filename = filename
        self.mode = mode

    #--------------------------------------------------------------------------
    # Main functions
    #--------------------------------------------------------------------------

    def read(self, path):
        """Read an HDF5 dataset, given its HDF5 path in the file."""
        return self._h5py_file[path]

    def read_attr(self, path, attr_name):
        """Read an attribute of an HDF5 group."""
        return self._h5py_file[path].attrs[attr_name]

    #--------------------------------------------------------------------------
    # Context manager
    #--------------------------------------------------------------------------

    def __enter__(self):
        self._h5py_file = h5py.File(self.filename, self.mode)
        return self

    def __exit__(self, type, value, tb):
        self._h5py_file.close()

    #--------------------------------------------------------------------------
    # Miscellaneous properties
    #--------------------------------------------------------------------------

    @property
    def h5py_file(self):
        """Native h5py file handle."""
        return self._h5py_file


def open_h5(filename, mode=None):
    return File(filename, mode=mode)
