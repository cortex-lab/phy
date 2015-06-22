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


def _dat_n_samples(filename, n_bits=None, n_channels=None):
    assert n_bits > 0
    n_samples = op.getsize(filename) // ((n_bits // 8) * n_channels)
    assert n_samples >= 0
    return n_samples


def read_ns5(filename):
    # TODO
    raise NotImplementedError()
