# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.tempdir import TemporaryDirectory
from ....utils.logging import set_level
from ..store import MemoryStore, DiskStore, Store, ClusterStore, StoreItem
from .._utils import _spikes_per_cluster
from .._update_info import UpdateInfo


#------------------------------------------------------------------------------
# Test data stores
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


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
        # Loading a non-existing key returns None.
        assert ds.load(3, 'key_bis') is None
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


def test_store_0():
    with TemporaryDirectory() as tempdir:
        cs = Store(tempdir)

        model = {'spike_clusters': np.random.randint(size=100, low=0, high=5)}

        def reset(model):
            cs.clear()
            # Find unique clusters.
            clusters = np.unique(model['spike_clusters'])
            # Load data for all clusters.
            generate(clusters)
            ae(cs.clusters('all'), clusters)

        def generate(clusters):
            ae(clusters, np.arange(5))

            for cluster in clusters:
                cs.store(cluster,
                         data_memory=np.array([1, 2]),
                         location='memory')

            # Test clusters() method.
            ae(cs.clusters('memory'), clusters)
            ae(cs.clusters('disk'), [])
            ae(cs.clusters('any'), clusters)
            ae(cs.clusters('all'), [])

            with raises(ValueError):
                cs.clusters('')
            with raises(ValueError):
                cs.clusters(None)

            for cluster in clusters:
                cs.store(cluster,
                         data_disk=np.array([3, 4]),
                         location='disk')

            # Test clusters() method.
            ae(cs.clusters('memory'), clusters)
            ae(cs.clusters('disk'), clusters)
            ae(cs.clusters('any'), clusters)
            ae(cs.clusters('all'), clusters)

        reset(model)
        ae(cs.load(3, 'data_memory'), [1, 2])
        ae(cs.load(4, 'data_disk'), [3, 4])


def test_cluster_store_1():
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
            name = 'my item'
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
        spc = spikes_per_cluster.copy()
        spikes = np.sort(np.concatenate([spc[0], spc[1]]))
        spc[20] = spikes
        del spc[0]
        del spc[1]
        up = UpdateInfo(added=[20], deleted=[0, 1],
                        spikes=spikes,
                        new_spikes_per_cluster=spc,
                        old_spikes_per_cluster=spikes_per_cluster,)

        cs.merge(up)

        # Check the list of clusters in the store.
        ae(cs._store.clusters('memory'), list(range(n_clusters)) + [20])
        ae(cs._store.clusters('disk'), [])
        assert cs.n_spikes(20) == len(spikes)

        # Recreate the cluster store.
        cs = ClusterStore(model=model, path=tempdir)
        cs.register_item(MyItem)
        cs.generate(spikes_per_cluster)
        ae(cs._store.clusters('memory'), list(range(n_clusters)))
        ae(cs._store.clusters('disk'), [])


def test_cluster_store_multi():
    """This tests the cluster store when a store item has several fields."""

    cs = ClusterStore()

    class MyItem(StoreItem):
        name = 'my item'
        fields = [('d', 'memory'),
                  ('m', 'memory')]

        def store_from_model(self, cluster, spikes):
            self.store.store(cluster, d=len(spikes), m=len(spikes)**2)

    cs.register_item(MyItem)

    cs.generate({0: [0, 2], 1: [1, 3, 4]})

    assert cs._store.load(0, ['d', 'm']) == {'d': 2, 'm': 4}
    assert cs.d(0) == 2
    assert cs.m(0) == 4

    assert cs._store.load(1, ['d', 'm']) == {'d': 3, 'm': 9}
    assert cs.d(1) == 3
    assert cs.m(1) == 9


def test_cluster_store_load():
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
            name = 'my item'
            fields = [('spikes_square', 'disk')]

            def store_from_model(self, cluster, spikes):
                self.store.store(cluster, spikes_square=spikes ** 2)

        cs.register_item(MyItem)

        # Now we generate the store.
        cs.generate(spikes_per_cluster)

        # All spikes in cluster 1.
        cluster = 1
        spikes = spikes_per_cluster[cluster]
        ae(cs.load('spikes_square', [cluster], spikes), spikes ** 2)

        # Some spikes in cluster 1.
        spikes = spikes_per_cluster[cluster][1::2]
        ae(cs.load('spikes_square', [cluster], spikes), spikes ** 2)

        # All spikes in several clusters.
        clusters = [2, 3, 5]
        spikes = np.concatenate([spikes_per_cluster[cl]
                                 for cl in clusters])
        # Reverse the order of spikes.
        spikes = np.r_[spikes, spikes[::-1]]
        ae(cs.load('spikes_square', clusters, spikes), spikes ** 2)

        # Some spikes in several clusters.
        spikes = np.concatenate([spikes_per_cluster[cl][::3]
                                 for cl in clusters])
        ae(cs.load('spikes_square', clusters, spikes), spikes ** 2)
