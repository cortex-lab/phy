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
        self._file_handle = h5py.File(self.filename, self.mode)
        return self._file_handle

    def __exit__(self, type, value, tb):
        self._file_handle.close()


def open_h5(filename, mode=None):
    return File(filename, mode=mode)
