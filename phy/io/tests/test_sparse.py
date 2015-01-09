# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from pytest import raises

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
    """Sparse matrix example:

        *   1   2   *   *
        3   *   *   4   *
        *   *   *   *   *
        *   *   5   *   *

    Return a dense array.

    """
    arr = np.zeros((4, 5))
    arr[0, 1] = 1
    arr[0, 2] = 2
    arr[1, 0] = 3
    arr[1, 3] = 4
    arr[3, 2] = 5
    return arr


def _sparse_matrix_example():
    """Return a sparse representation of the sparse matrix example."""
    data = np.arange(1, 6)
    channels = np.array([1, 2, 0, 3, 2])
    spikes_ptr = np.array([0, 2, 4, 5])
    return data, channels, spikes_ptr


def test_sparse_csr_check():
    """Test the checks performed when creating a sparse matrix."""
    arr = _dense_matrix_example()
    with raises(NotImplementedError):
        csr_matrix(arr)

    data_exp = np.arange(1, 6)

    with raises(ValueError):
        csr_matrix(data=data_exp)
    with raises(ValueError):
        csr_matrix(channels=None)
    with raises(ValueError):
        csr_matrix(data=data_exp)
