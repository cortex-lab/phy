# -*- coding: utf-8 -*-

"""Raw data readers."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np


#------------------------------------------------------------------------------
# Raw data readers
#------------------------------------------------------------------------------

def read_dat(filename, dtype=None, shape=None, offset=0):
    return np.memmap(filename, dtype=dtype, shape=shape,
                     mode='r', offset=offset)


def _dat_n_samples(filename, dtype=None, n_channels=None):
    assert dtype
    item_size = np.dtype(dtype).itemsize
    n_samples = op.getsize(filename) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def read_ns5(filename):
    # TODO
    raise NotImplementedError()
