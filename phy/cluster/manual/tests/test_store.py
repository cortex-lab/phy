# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.logging import set_level
from ....utils.tempdir import TemporaryDirectory
from ..store import MemoryStore, DiskStore, BaseClusterStore


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
    assert ms.clusters == []

    ms.store(3, key='a')
    assert ms.load(3) == {'key': 'a'}
    assert ms.load(3, ['key']) == {'key': 'a'}
    assert ms.load(3, 'key') == 'a'
    assert ms.clusters == [3]

    ms.store(3, key_bis='b')
    assert ms.load(3) == {'key': 'a', 'key_bis': 'b'}
    assert ms.load(3, ['key']) == {'key': 'a'}
    assert ms.load(3, ['key_bis']) == {'key_bis': 'b'}
    assert ms.load(3, ['key', 'key_bis']) == {'key': 'a', 'key_bis': 'b'}
    assert ms.load(3, 'key_bis') == 'b'
    assert ms.clusters == [3]

    ms.delete([2, 3])
    assert ms.load(3) == {}
    assert ms.load(3, ['key']) == {'key': None}
    assert ms.clusters == []


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
        assert ds.clusters == []

        ds.store(3, key=a)
        _assert_equal(ds.load(3), {'key': a})
        _assert_equal(ds.load(3, ['key']), {'key': a})
        ae(ds.load(3, 'key'), a)
        assert ds.clusters == [3]

        ds.store(3, key_bis=b)
        _assert_equal(ds.load(3), {'key': a, 'key_bis': b})
        _assert_equal(ds.load(3, ['key']), {'key': a})
        _assert_equal(ds.load(3, ['key_bis']), {'key_bis': b})
        _assert_equal(ds.load(3, ['key', 'key_bis']), {'key': a, 'key_bis': b})
        ae(ds.load(3, 'key_bis'), b)
        assert ds.clusters == [3]

        ds.delete([2, 3])
        assert ds.load(3) == {}
        assert ds.load(3, ['key']) == {'key': None}
        assert ds.clusters == []


def test_cluster_store():
    with TemporaryDirectory() as tempdir:
        cs = BaseClusterStore('test', root_path=tempdir)

        model = {'spike_clusters': np.random.randint(size=100, low=0, high=10)}

        @cs.connect
        def on_reset(model):
            cs.clear()
            # Find unique clusters.
            clusters = np.unique(model['spike_clusters'])
            # Load data for all clusters.
            cs.generate(clusters)
            ae(cs.clusters, clusters)

        @cs.connect
        def on_generate(clusters):
            for cluster in clusters:

                cs.store(cluster,
                         data_memory=np.array([1, 2]),
                         location='memory')

                cs.store(cluster,
                         data_disk=np.array([3, 4]),
                         location='disk')

        cs.reset(model)
        ae(cs.load(3, 'data_memory'), [1, 2])
        ae(cs.load(5, 'data_disk'), [3, 4])
