# -*- coding: utf-8 -*-

"""Test context."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import yield_fixture
from six.moves import cPickle

from ..array import write_array, read_array
from ..context import Context, _fullname


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture(scope='function')
def context(tempdir):
    ctx = Context('{}/cache/'.format(tempdir), verbose=1)
    yield ctx


@yield_fixture
def temp_phy_config_dir(tempdir):
    """Use a temporary phy user directory."""
    import phy.io.context
    f = phy.io.context.phy_config_dir
    phy.io.context.phy_config_dir = lambda: tempdir
    yield
    phy.io.context.phy_config_dir = f


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


def test_context_load_save(tempdir, context, temp_phy_config_dir):
    assert not context.load('unexisting')

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
    with open(op.join(tempdir, 'test.pkl'), 'wb') as f:
        cPickle.dump(context, f)
    with open(op.join(tempdir, 'test.pkl'), 'rb') as f:
        ctx = cPickle.load(f)
    assert isinstance(ctx, Context)
    assert ctx.cache_dir == context.cache_dir
