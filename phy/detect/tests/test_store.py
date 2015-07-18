# -*- coding: utf-8 -*-

"""Tests of spike detection store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture
import numpy as np
from numpy.testing import assert_equal as ae

from ...utils.logging import set_level
from ..store import SpikeCounts, ArrayStore


#------------------------------------------------------------------------------
# Tests spike detection store
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


@yield_fixture(params=['from_dict', 'append'])
def spike_counts(request):
    groups = [0, 2]
    chunk_keys = [10, 20, 30]
    if request.param == 'from_dict':
        c = {0: {10: 100, 20: 200},
             2: {10: 1, 30: 300},
             }
        sc = SpikeCounts(c, groups=groups, chunk_keys=chunk_keys)
    elif request.param == 'append':
        sc = SpikeCounts(groups=groups, chunk_keys=chunk_keys)
        sc.append(group=0, chunk_key=10, count=100)
        sc.append(group=0, chunk_key=20, count=200)
        sc.append(group=2, chunk_key=10, count=1)
        sc.append(group=2, chunk_key=30, count=300)
    yield sc


def test_spike_counts(spike_counts):
    assert spike_counts() == 601

    assert spike_counts(group=0) == 300
    assert spike_counts(group=1) == 0
    assert spike_counts(group=2) == 301

    assert spike_counts(chunk_key=10) == 101
    assert spike_counts(chunk_key=20) == 200
    assert spike_counts(chunk_key=30) == 300


class TestArrayStore(ArrayStore):
    def _rel_path(self, a=None, b=None):
        return '{}/{}.npy'.format(a or 'none_a', b or 'none_b')


def test_array_store(tempdir):
    store = TestArrayStore(tempdir)

    store.store(a=1, b=1, data=np.arange(11))
    store.store(a=1, b=2, data=np.arange(12))
    store.store(b=2, data=np.arange(2))
    store.store(b=3, data=None)
    store.store(a=1, data=[np.arange(3), np.arange(3, 8)])

    ae(store.load(a=0, b=1), None)
    ae(store.load(a=1, b=1), np.arange(11))
    ae(store.load(a=1, b=2), np.arange(12))
    ae(store.load(b=2), np.arange(2))
    ae(store.load(b=3), None)
    ae(store.load(a=1)[0], np.arange(3))
    ae(store.load(a=1)[1], np.arange(3, 8))
