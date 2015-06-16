# -*- coding: utf-8 -*-

"""Raw data readers."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Raw data readers
#------------------------------------------------------------------------------

def read_dat(filename, dtype=None, shape=None, offset=0):
    return np.memmap(filename, dtype=dtype, shape=shape,
                     mode='r', offset=offset)


def read_ns5(filename):
    # TODO
    raise NotImplementedError()
