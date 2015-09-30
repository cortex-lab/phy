# -*- coding: utf-8 -*-

"""Test context."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import yield_fixture

from ..context import Context, _iter_chunks_dask


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def context(tempdir):
    ctx = Context('{}/cache/'.format(tempdir))
    yield ctx


@yield_fixture(scope='module')
def ipy_client():

    def iptest_stdstreams_fileno():
        return os.open(os.devnull, os.O_WRONLY)

    # OMG-THIS-IS-UGLY-HACK: monkey-patch this global object to avoid
    # using the nose iptest extension (we're using pytest).
    # See https://github.com/ipython/ipython/blob/master/IPython/testing/iptest.py#L317-L319  # noqa
    from ipyparallel import Client
    import ipyparallel.tests
    ipyparallel.tests.nose.iptest_stdstreams_fileno = iptest_stdstreams_fileno

    # Start two engines engine (one is launched by setup()).
    ipyparallel.tests.setup()
    ipyparallel.tests.add_engines(1)
    yield Client(profile='iptest')
    ipyparallel.tests.teardown()


#------------------------------------------------------------------------------
# ipyparallel tests
#------------------------------------------------------------------------------

def test_client_1(ipy_client):
    assert ipy_client.ids == [0, 1]


def test_client_2(ipy_client):
    assert ipy_client[:].map_sync(lambda x: x * x, [1, 2, 3]) == [1, 4, 9]


#------------------------------------------------------------------------------
# Test context
#------------------------------------------------------------------------------

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


def test_context_parallel_map(context, ipy_client):
    view = ipy_client[:]
    context.ipy_view = view
    assert context.ipy_view == view

    def square(x):
        return x * x

    assert context.map(square, [1, 2, 3]) == [1, 4, 9]
    assert context.map_async(square, [1, 2, 3]).get() == [1, 4, 9]


def test_context_parallel_dask(context, ipy_client):
    from dask.array import from_array

    context.ipy_view = ipy_client[:]

    def square(x):
        import os
        print(os.getpid())
        return x * x

    x = np.arange(10)
    da = from_array(x, chunks=(3,))
    res = context.map_dask_array(square, da)
    ae(res.compute(), x ** 2)
