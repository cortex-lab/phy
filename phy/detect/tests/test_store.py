# -*- coding: utf-8 -*-

"""Tests of spike detection store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_equal as ae

from ...utils.logging import set_level
from ..store import ArrayStore


#------------------------------------------------------------------------------
# Tests spike detection store
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


class TestArrayStore(ArrayStore):
    def _rel_path(self, a=None, b=None):
        return '{}/{}.npy'.format(a or 'none_a', b or 'none_b')


def test_array_store(tempdir):
    store = TestArrayStore(tempdir)

    store.store(a=1, b=1, data=np.arange(11))
    store.store(a=1, b=2, data=np.arange(12))
    store.store(b=2, data=np.arange(2))
    store.store(b=3, data=None)
    store.store(a=1, data=np.arange(10))

    ae(store.load(a=0, b=1), None)
    ae(store.load(a=1, b=1), np.arange(11))
    ae(store.load(a=1, b=2), np.arange(12))
    ae(store.load(b=2), np.arange(2))
    ae(store.load(b=3), None)
    ae(store.load(a=1), np.arange(10))
