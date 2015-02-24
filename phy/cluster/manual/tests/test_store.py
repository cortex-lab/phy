# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.logging import set_level
from ....utils.tempdir import TemporaryDirectory
from ..store import MemoryStore, DiskStore


#------------------------------------------------------------------------------
# Test data stores
#------------------------------------------------------------------------------

def test_memory_store():
    ms = MemoryStore()
    assert ms.load(2) == {}

    assert ms.load(3).get('key', None) is None
    assert ms.load(3) == {}
    assert ms.load(3, ['key']) == {'key': None}
    assert ms.load(3) == {}
    assert ms.keys() == []

    ms.store(3, key='a')
    assert ms.load(3) == {'key': 'a'}
    assert ms.load(3, ['key']) == {'key': 'a'}
    assert ms.load(3, 'key') == 'a'
    assert ms.keys() == [3]

    ms.store(3, key_bis='b')
    assert ms.load(3) == {'key': 'a', 'key_bis': 'b'}
    assert ms.load(3, ['key']) == {'key': 'a'}
    assert ms.load(3, ['key_bis']) == {'key_bis': 'b'}
    assert ms.load(3, ['key', 'key_bis']) == {'key': 'a', 'key_bis': 'b'}
    assert ms.load(3, 'key_bis') == 'b'
    assert ms.keys() == [3]

    ms.delete([2, 3])
    assert ms.load(3) == {}
    assert ms.load(3, ['key']) == {'key': None}
    assert ms.keys() == []


def test_disk_store():

    a = np.random.rand(2, 4)
    b = np.random.rand(3, 5)

    def _assert_equal(d_0, d_1):
        """Test the equality of two dictionaries containing NumPy arrays."""
        assert sorted(d_0.keys()) == sorted(d_1.keys())
        for key in d_0.keys():
            ae(d_0[key], d_1[key])

    with TemporaryDirectory() as tempdir:
        ds = DiskStore(tempdir)

        assert ds.load(2) == {}

        assert ds.load(3).get('key', None) is None
        assert ds.load(3) == {}
        assert ds.load(3, ['key']) == {'key': None}
        assert ds.load(3) == {}
        assert ds.keys() == []

        ds.store(3, key=a)
        _assert_equal(ds.load(3), {'key': a})
        _assert_equal(ds.load(3, ['key']), {'key': a})
        ae(ds.load(3, 'key'), a)
        assert ds.keys() == [3]

        ds.store(3, key_bis=b)
        _assert_equal(ds.load(3), {'key': a, 'key_bis': b})
        _assert_equal(ds.load(3, ['key']), {'key': a})
        _assert_equal(ds.load(3, ['key_bis']), {'key_bis': b})
        _assert_equal(ds.load(3, ['key', 'key_bis']), {'key': a, 'key_bis': b})
        ae(ds.load(3, 'key_bis'), b)
        assert ds.keys() == [3]

        ds.delete([2, 3])
        assert ds.load(3) == {}
        assert ds.load(3, ['key']) == {'key': None}
        assert ds.keys() == []
