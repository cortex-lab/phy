# -*- coding: utf-8 -*-

"""Tests of read traces functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from numpy.testing import assert_array_equal as ae

from ..traces import read_dat, _dat_n_samples
from ..mock import artificial_traces
from ...utils.tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_read_dat():
    n_samples = 100
    n_channels = 10

    arr = artificial_traces(n_samples, n_channels)

    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')
        arr.tofile(path)
        assert _dat_n_samples(path, n_bits=64,
                              n_channels=n_channels) == n_samples
        data = read_dat(path, dtype=arr.dtype, shape=arr.shape)

    ae(arr, data)
