# -*- coding: utf-8 -*-

"""Test context."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import fixture, yield_fixture, mark
from ipyparallel.tests.clienttest import ClusterTestCase, add_engines

from ..context import Context, _iter_chunks_dask


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def context(tempdir):
    ctx = Context('{}/cache/'.format(tempdir))
    yield ctx


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


#------------------------------------------------------------------------------
# ipyparallel tests
#------------------------------------------------------------------------------

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


def test_client(ipy_client):
    print(ipy_client.ids)
