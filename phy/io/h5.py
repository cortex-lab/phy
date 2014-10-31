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
# HDF5 functions
#------------------------------------------------------------------------------

def open_h5(file, mode=None):
    if mode is None:
        mode = 'r'
    # TODO
