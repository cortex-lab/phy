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

    def __enter__(self):
        self._h5py_file = h5py.File(self.filename, self.mode)
        return self

    def __exit__(self, type, value, tb):
        self._h5py_file.close()

    @property
    def h5py_file(self):
        return self._h5py_file


def open_h5(filename, mode=None):
    return File(filename, mode=mode)
