# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ....utils.tempdir import TemporaryDirectory
from ..store import MemoryStore, DiskStore


#------------------------------------------------------------------------------
# Test data stores
#------------------------------------------------------------------------------

def _test_store(ms):
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


def test_memory_store():
    ms = MemoryStore()
    _test_store(ms)


def test_disk_store():
    with TemporaryDirectory() as tempdir:
        ds = DiskStore(tempdir)
        assert ds  # _test_store(ds)
