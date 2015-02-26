# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.logging import set_level
from ....utils.tempdir import TemporaryDirectory
from ..store import MemoryStore, DiskStore, Store, ClusterStore, StoreItem
from .._utils import _spikes_per_cluster
from .._update_info import UpdateInfo


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


def test_store():
    with TemporaryDirectory() as tempdir:
        cs = Store(tempdir)

        model = {'spike_clusters': np.random.randint(size=100, low=0, high=10)}

        def reset(model):
            cs.clear()
            # Find unique clusters.
            clusters = np.unique(model['spike_clusters'])
            # Load data for all clusters.
            generate(clusters)
            ae(cs.clusters, clusters)

        def generate(clusters):
            for cluster in clusters:

                cs.store(cluster,
                         data_memory=np.array([1, 2]),
                         location='memory')

                cs.store(cluster,
                         data_disk=np.array([3, 4]),
                         location='disk')

        reset(model)
        ae(cs.load(3, 'data_memory'), [1, 2])
        ae(cs.load(5, 'data_disk'), [3, 4])


def test_cluster_store():
    with TemporaryDirectory() as tempdir:

        # We define some data and a model.
        n_spikes = 100
        n_clusters = 10

        spike_ids = np.arange(n_spikes)
        spike_clusters = np.random.randint(size=n_spikes,
                                           low=0, high=n_clusters)
        spikes_per_cluster = _spikes_per_cluster(spike_ids, spike_clusters)

        model = {'spike_clusters': spike_clusters}

        # We initialize the ClusterStore.
        cs = ClusterStore(model=model, path=tempdir)

        # We create a n_spikes item to be stored in memory,
        # and we define how to generate it for a given cluster.
        class MyItem(StoreItem):
            fields = [('n_spikes', 'memory')]

            def store_from_model(self, cluster, spikes):
                self.store.store(cluster, n_spikes=len(spikes))

            def merge(self, up):
                n = sum(len(up.old_spikes_per_cluster[cl])
                        for cl in up.deleted)
                self.store.store(up.added[0], n_spikes=n)

        cs.register_item(MyItem)

        # Now we generate the store.
        cs.generate(spikes_per_cluster)

        # We check that the n_spikes field has successfully been created.
        for cluster in sorted(spikes_per_cluster):
            assert cs.n_spikes(cluster) == len(spikes_per_cluster[cluster])

        # Merge.
        spc = spikes_per_cluster
        spikes = np.sort(np.concatenate([spc[0], spc[1]]))
        spc[20] = spikes
        up = UpdateInfo(added=[20], deleted=[0, 1],
                        spikes=spikes,
                        new_spikes_per_cluster=spc,
                        old_spikes_per_cluster=spc,)

        cs.merge(up)
        assert cs.n_spikes(20) == len(spikes)
