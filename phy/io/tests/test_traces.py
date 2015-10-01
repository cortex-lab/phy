# -*- coding: utf-8 -*-

"""Tests of read traces functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import raises

from ..traces import read_dat, _dat_n_samples, read_kwd, read_ns5
from ..mock import artificial_traces


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_read_dat(tempdir):
    n_samples = 100
    n_channels = 10

    arr = artificial_traces(n_samples, n_channels)

    path = op.join(tempdir, 'test')
    arr.tofile(path)
    assert _dat_n_samples(path, dtype=np.float64,
                          n_channels=n_channels) == n_samples
    data = read_dat(path, dtype=arr.dtype, shape=arr.shape)
    ae(arr, data)
    data = read_dat(path, dtype=arr.dtype, n_channels=n_channels)
    ae(arr, data)


def test_read_kwd(tempdir):
    from h5py import File

    n_samples = 100
    n_channels = 10
    arr = artificial_traces(n_samples, n_channels)
    path = op.join(tempdir, 'test.kwd')

    with File(path, 'w') as f:
        g0 = f.create_group('/recordings/0')
        g1 = f.create_group('/recordings/1')

        arr0 = arr[:n_samples // 2, ...]
        arr1 = arr[n_samples // 2:, ...]

        g0.create_dataset('data', data=arr0)
        g1.create_dataset('data', data=arr1)

    ae(read_kwd(path)[...], arr)


def test_read_ns5():
    with raises(NotImplementedError):
        read_ns5('')
