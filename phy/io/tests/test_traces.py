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

from ..h5 import open_h5
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
    n_samples = 100
    n_channels = 10

    arr = artificial_traces(n_samples, n_channels)

    path = op.join(tempdir, 'test')

    with open_h5(path, 'w') as f:
        f.write('/recordings/0/data',
                arr[:n_samples // 2, ...].astype(np.float32))
        f.write('/recordings/1/data',
                arr[n_samples // 2:, ...].astype(np.float32))

    with open_h5(path, 'r') as f:
        data = read_kwd(f)[...]
        ac(arr, data)


def test_read_ns5():
    with raises(NotImplementedError):
        read_ns5('')
