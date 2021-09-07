# -*- coding: utf-8 -*-

"""Test context."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pickle import dump, load

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import fixture

from phylib.io.array import write_array, read_array
from ..context import Context, _fullname


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture(scope='function')
def context(tempdir):
    ctx = Context('{}/cache/'.format(tempdir), verbose=1)
    return ctx


@fixture
def temp_phy_config_dir(tempdir):
    """Use a temporary phy user directory."""
    import phy.utils.context
    f = phy.utils.context.phy_config_dir
    phy.utils.context.phy_config_dir = lambda: tempdir
    yield
    phy.utils.context.phy_config_dir = f


#------------------------------------------------------------------------------
# Test utils and cache
#------------------------------------------------------------------------------

def test_read_write(tempdir):
    x = np.arange(10)
    write_array(tempdir / 'test.npy', x)
    ae(read_array(tempdir / 'test.npy'), x)


def test_context_load_save(tempdir, context, temp_phy_config_dir):
    assert not context.load('unexisting')

    context.save('a/hello', {'text': 'world'})
    assert context.load('a/hello')['text'] == 'world'

    context.save('a/hello', {'text': 'world!'}, location='global')
    assert context.load('a/hello', location='global')['text'] == 'world!'


def test_context_load_save_pickle(tempdir, context, temp_phy_config_dir):
    arr = np.random.rand(10, 10)
    context.save('arr', arr, kind='pickle')
    ae(context.load('arr'), arr)


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


def test_context_cache_method(tempdir, context):
    class A(object):
        def __init__(self, ctx):
            self.f = ctx.cache(self.f)
            self._l = []

        def f(self, x):
            self._l.append(x)
            return x

    a = A(context)
    assert not a._l

    # First call: the function is executed.
    assert a.f(3) == 3
    assert a._l == [3]

    # Second call: the function is not executed.
    assert a.f(3) == 3
    assert a._l == [3]

    # Recreate the context.
    context = Context('{}/cache/'.format(tempdir), verbose=1)
    # Recreate the class.
    a = A(context)
    assert a.f(3) == 3
    # The function is not called after reinitialization of the object.
    assert not a._l


def test_context_memcache(tempdir, context):

    _res = []

    @context.memcache
    def f(x):
        _res.append(x)
        return x ** 2

    # Compute the function a first time.
    x = 10
    ae(f(x), x ** 2)
    assert len(_res) == 1

    # The second time, the memory cache is used.
    ae(f(x), x ** 2)
    assert len(_res) == 1

    # We artificially clear the memory cache.
    context.save_memcache()
    del context._memcache[_fullname(f)]
    context.load_memcache(_fullname(f))

    # This time, the result is loaded from disk.
    ae(f(x), x ** 2)
    assert len(_res) == 1


def test_pickle_cache(tempdir, context):
    """Make sure the Context is picklable."""
    with open(tempdir / 'test.pkl', 'wb') as f:
        dump(context, f)
    with open(tempdir / 'test.pkl', 'rb') as f:
        ctx = load(f)
    assert isinstance(ctx, Context)
    assert ctx.cache_dir == context.cache_dir
