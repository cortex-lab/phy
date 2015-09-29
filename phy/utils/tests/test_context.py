# -*- coding: utf-8 -*-

"""Test context."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import yield_fixture

from ..context import Context, _iter_chunks_dask


#------------------------------------------------------------------------------
# Test context
#------------------------------------------------------------------------------

@yield_fixture
def context(tempdir):
    ctx = Context('{}/cache/'.format(tempdir))
    yield ctx


def test_iter_chunks_dask():
    from dask.array import from_array

    x = np.arange(10)
    da = from_array(x, chunks=(3,))
    assert len(list(_iter_chunks_dask(da))) == 4


def test_context_map(context):
    def f(x):
        return x * x

    args = range(10)
    assert context.map(f, args) == [x * x for x in range(10)]


def test_context_dask(context):
    from dask.array import from_array

    def square(x):
        return x * x

    x = np.arange(10)
    da = from_array(x, chunks=(3,))
    res = context.map_dask_array(square, da)
    ae(res.compute(), x ** 2)
