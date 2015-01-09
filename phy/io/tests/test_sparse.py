# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from nose.tools import assert_raises

from ..sparse import csr_matrix


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    pass


def teardown():
    pass


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------


def _dense_matrix_example():
    arr = np.zeros((4, 5))
    arr[0, 1] = 1
    arr[0, 2] = 2
    arr[1, 0] = 3
    arr[1, 3] = 4
    arr[3, 4] = 5
    return arr


def test_sparse_csr_check():
    arr = _dense_matrix_example()
    assert_raises(NotImplementedError, csr_matrix, (arr,))

    data_exp = np.arange(1, 6)

    data = data_exp
    channels = None
    spikes_ptr = None
    csr_matrix(data=data, channels=channels, spikes_ptr=spikes_ptr)
