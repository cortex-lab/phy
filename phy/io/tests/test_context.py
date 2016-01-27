# -*- coding: utf-8 -*-

"""Test context."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import shutil

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import yield_fixture, mark, raises
from six.moves import cPickle

from ..context import (Context, Task,
                       _iter_chunks_dask, write_array, read_array,
                       _fullname,
                       )


#------------------------------------------------------------------------------
# Fixtures
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


@yield_fixture(scope='function')
def context(tempdir):
    ctx = Context('{}/cache/'.format(tempdir), verbose=1)
    yield ctx


@yield_fixture(scope='function', params=[False, True])
def parallel_context(tempdir, ipy_client, request):
    """Parallel and non-parallel context."""
    ctx = Context('{}/cache/'.format(tempdir))
    if request.param:
        ctx.ipy_view = ipy_client[:]
    yield ctx


@yield_fixture
def temp_phy_user_dir(tempdir):
    """Use a temporary phy user directory."""
    import phy.io.context
    f = phy.io.context.phy_user_dir
    phy.io.context.phy_user_dir = lambda: tempdir
    yield
    phy.io.context.phy_user_dir = f


#------------------------------------------------------------------------------
# ipyparallel tests
#------------------------------------------------------------------------------

def test_client_1(ipy_client):
    assert ipy_client.ids == [0, 1]
    assert ipy_client[:].map_sync(lambda x: x * x, [1, 2, 3]) == [1, 4, 9]


#------------------------------------------------------------------------------
# Test utils and cache
#------------------------------------------------------------------------------

def test_fullname():
    def myfunction(x):
        return x

    assert _fullname(myfunction) == 'phy.io.tests.test_context.myfunction'


def test_read_write(tempdir):
    x = np.arange(10)
    write_array(op.join(tempdir, 'test.npy'), x)
    ae(read_array(op.join(tempdir, 'test.npy')), x)


def test_context_load_save(tempdir, context, temp_phy_user_dir):
    assert context.load('unexisting') is None

    context.save('a/hello', {'text': 'world'})
    assert context.load('a/hello')['text'] == 'world'

    context.save('a/hello', {'text': 'world!'}, location='global')
    assert context.load('a/hello', location='global')['text'] == 'world!'


def test_context_cache(context):

    _res = []

    def f(x):
        _res.append(x)
        return x ** 2

    x = np.arange(5)
    x2 = x * x

    ae(f(x), x2)
    assert len(_res) == 1

    f = context.cache(f)

    # Run it a first time.
    ae(f(x), x2)
    assert len(_res) == 2

    # The second time, the cache is used.
    ae(f(x), x2)
    assert len(_res) == 2


def test_context_memcache(tempdir, context):

    _res = []

    @context.memcache
    @context.cache
    def f(x):
        _res.append(x)
        return x ** 2

    # Compute the function a first time.
    x = np.arange(10)
    ae(f(x), x ** 2)
    assert len(_res) == 1

    # The second time, the memory cache is used.
    ae(f(x), x ** 2)
    assert len(_res) == 1

    # We artificially clear the memory cache.
    context._memcache[_fullname(f)].clear()

    # This time, the result is loaded from disk.
    ae(f(x), x ** 2)
    assert len(_res) == 1

    # Remove the cache directory.
    assert context.cache_dir.replace('/private', '').startswith(tempdir)
    shutil.rmtree(context.cache_dir)
    context._memcache[_fullname(f)].clear()

    # Now, the result is re-computed.
    ae(f(x), x ** 2)
    assert len(_res) == 2


def test_pickle_cache(tempdir, parallel_context):
    """Make sure the Context is picklable."""
    with open(op.join(tempdir, 'test.pkl'), 'wb') as f:
        cPickle.dump(parallel_context, f)
    with open(op.join(tempdir, 'test.pkl'), 'rb') as f:
        ctx = cPickle.load(f)
    assert isinstance(ctx, Context)
    assert ctx.cache_dir == parallel_context.cache_dir


#------------------------------------------------------------------------------
# Test map
#------------------------------------------------------------------------------

def test_context_map(parallel_context):

    def square(x):
        return x * x

    assert parallel_context.map(square, [1, 2, 3]) == [1, 4, 9]
    if not parallel_context.ipy_view:
        with raises(RuntimeError):
            parallel_context.map_async(square, [1, 2, 3])
    else:
        assert parallel_context.map_async(square, [1, 2, 3]).get() == [1, 4, 9]


def test_task():
    task = Task(ctx=None)
    assert task


#------------------------------------------------------------------------------
# Test context dask
#------------------------------------------------------------------------------

def test_iter_chunks_dask():
    from dask.array import from_array

    x = np.arange(10)
    da = from_array(x, chunks=(3,))
    assert len(list(_iter_chunks_dask(da))) == 4


@mark.parametrize('multiple_outputs', [True, False])
def test_context_dask(parallel_context, multiple_outputs):
    from dask.array import from_array, from_npy_stack
    context = parallel_context

    if not multiple_outputs:
        def f4(x, onset):
            return x * x * x * x
        name = None
    else:
        def f4(x, onset):
            return x * x * x * x + onset, x + 1
        name = ('power_four', 'plus_one')

    x = np.arange(10)
    da = from_array(x, chunks=(3,))
    res = context.map_dask_array(f4, da, 0, name=name)

    # Check that we can load the dumped dask array from disk.
    # The location is in the context cache dir, in a subdirectory with the
    # name of the function by default.
    if not multiple_outputs:
        ae(res.compute(), x ** 4)

        y = from_npy_stack(op.join(context.cache_dir, 'f4'))
        ae(y.compute(), x ** 4)
    else:
        ae(res[0].compute(), x ** 4)
        ae(res[1].compute(), x + 1)
        y = from_npy_stack(op.join(context.cache_dir, 'power_four'))
        ae(y.compute(), x ** 4)

        y = from_npy_stack(op.join(context.cache_dir, 'plus_one'))
        ae(y.compute(), x + 1)
