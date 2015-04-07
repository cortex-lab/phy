# -*- coding: utf-8 -*-

"""Cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

import numpy as np

from ...utils.array import _is_array_like, _index_of
from ...utils.logging import debug
from ...io.h5 import open_h5
from ...ext.six import string_types


#------------------------------------------------------------------------------
# Data stores
#------------------------------------------------------------------------------

class MemoryStore(object):
    """Store cluster-related data in memory."""
    def __init__(self):
        self._ds = {}

    def store(self, cluster, **data):
        """Store cluster-related data."""
        if cluster not in self._ds:
            self._ds[cluster] = {}
        self._ds[cluster].update(data)

    def load(self, cluster, keys=None):
        """Load cluster-related data."""
        if keys is None:
            return self._ds.get(cluster, {})
        else:
            if isinstance(keys, string_types):
                return self._ds.get(cluster, {}).get(keys, None)
            assert isinstance(keys, (list, tuple))
            return {key: self._ds.get(cluster, {}).get(key, None)
                    for key in keys}

    @property
    def clusters(self):
        """List of cluster ids in the store."""
        return sorted(self._ds.keys())

    def delete(self, clusters):
        """Delete some clusters from the store."""
        assert isinstance(clusters, list)
        for cluster in clusters:
            if cluster in self._ds:
                del self._ds[cluster]

    def clear(self):
        """Clear the store completely by deleting all clusters."""
        self.delete(self.clusters)


class DiskStore(object):
    """Store cluster-related data in HDF5 files."""
    def __init__(self, directory):
        assert directory is not None
        self._directory = op.realpath(directory)

    # Internal methods
    # -------------------------------------------------------------------------

    def _cluster_path(self, cluster):
        """Return the absolute path of a cluster in the disk store."""
        # TODO: subfolders
        rel_path = '{0:05d}.h5'.format(cluster)
        return op.realpath(op.join(self._directory, rel_path))

    def _cluster_file_exists(self, cluster):
        """Return whether a cluster file exists."""
        return op.exists(self._cluster_path(cluster))

    def cluster_file(self, cluster, mode):
        """Return a file handle of a cluster file."""
        path = self._cluster_path(cluster)
        return open_h5(path, mode)

    def cluster_array(self, f, key):
        """Return an array from an already-open cluster file."""
        return self._get(f, key, return_ndarray=False)

    # Data get/set methods
    # -------------------------------------------------------------------------

    def _get(self, f, key, return_ndarray=True):
        """Return the data for a given key."""
        path = '/{0:s}'.format(key)
        if f.exists(path):
            arr = f.read(path)
            if return_ndarray:
                arr = arr[...]
            return arr
        else:
            return None

    def _set(self, f, key, value):
        """Set the data for a given key."""
        path = '/{0:s}'.format(key)
        f.write(path, value, overwrite=True)

    # Public methods
    # -------------------------------------------------------------------------

    def store(self, cluster, **data):
        """Store cluster-related data."""
        # Do not create the file if there's nothing to write.
        if not data:
            return
        with self.cluster_file(cluster, 'a') as f:
            for key, value in data.items():
                self._set(f, key, value)

    def load(self, cluster, keys=None):
        """Load cluster-related data."""
        # The cluster doesn't exist: return None for all keys.
        if not self._cluster_file_exists(cluster):
            if keys is None:
                return {}
            elif isinstance(keys, string_types):
                return None
            elif isinstance(keys, list):
                return {key: None for key in keys}
            else:
                raise ValueError(keys)
        # Create the output dictionary.
        out = {}
        # Open the cluster file in read mode.
        with self.cluster_file(cluster, 'r') as f:
            # If a single key is requested, return the value.
            if isinstance(keys, string_types):
                return self._get(f, keys)
            # All keys are requested if None.
            if keys is None:
                keys = f.datasets()
            assert isinstance(keys, (list, tuple))
            # Fetch the values for all requested keys.
            for key in keys:
                out[key] = self._get(f, key)
        return out

    @property
    def clusters(self):
        """List of cluster ids in the store."""
        if not op.exists(self._directory):
            return []
        files = os.listdir(self._directory)
        clusters = [int(op.splitext(file)[0]) for file in files]
        return sorted(clusters)

    def delete(self, clusters):
        """Delete some clusters from the store."""
        for cluster in clusters:
            if self._cluster_file_exists(cluster):
                os.remove(self._cluster_path(cluster))

    def clear(self):
        """Clear the store completely by deleting all clusters."""
        self.delete(self.clusters)


#------------------------------------------------------------------------------
# Cluster store
#------------------------------------------------------------------------------

class ClusterStore(object):
    def __init__(self, model=None, path=None):
        self._model = model
        self._spikes_per_cluster = {}
        self._memory = MemoryStore()
        self._disk = DiskStore(path) if path is not None else None
        self._items = []
        self._locations = {}

    def _store(self, location):
        if location == 'memory':
            return self._memory
        elif location == 'disk':
            return self._disk
        else:
            raise ValueError("The 'location' should be 'memory' "
                             "or 'disk'.")

    @property
    def memory_store(self):
        return self._memory

    @property
    def disk_store(self):
        return self._disk

    @property
    def spikes_per_cluster(self):
        return self._spikes_per_cluster

    def register_item(self, item_cls):
        """Register a StoreItem instance in the store."""

        # Instanciate the item.
        item = item_cls(model=self._model,
                        memory_store=self._memory,
                        disk_store=self._disk)
        assert item.fields is not None

        # HACK: need to use a factory function because in Python
        # functions are closed over names, not values. Here we
        # want 'name' to refer to the 'name' local variable.

        def _make_func(name, location):
            return lambda cluster: self._store(location).load(cluster, name)

        for name, location in item.fields:

            # Register the item location (memory or store).
            assert name not in self._locations
            self._locations[name] = location

            # Get the load function.
            load = _make_func(name, location)

            # We create the self.<name>(cluster) method for loading.
            # We need to ensure that the method name isn't already attributed.
            assert not hasattr(self, name)
            setattr(self, name, load)

        # Register the StoreItem instance.
        self._items.append(item)

    def load(self, name, clusters, spikes):
        assert _is_array_like(clusters)
        location = self._locations[name]
        store = self._store(location)

        # Concatenation of arrays for all clusters.
        arrays = np.concatenate([store.load(cluster, name)
                                 for cluster in clusters])
        # Concatenation of spike indices for all clusters.
        spike_clusters = np.concatenate([self._spikes_per_cluster[cluster]
                                         for cluster in clusters])
        assert np.all(np.in1d(spikes, spike_clusters))
        idx = _index_of(spikes, spike_clusters)
        return arrays[idx, ...]

    def update(self, up):
        # TODO: update self._spikes_per_cluster
        # Delete the deleted clusters from the store.
        self._memory.delete(up.deleted)
        self._disk.delete(up.deleted)

        if up.description == 'merge':
            self.merge(up)
        elif up.description == 'assign':
            self.assign(up)
        else:
            raise NotImplementedError()

    def merge(self, up):
        for item in self._items:
            item.merge(up)

    def assign(self, up):
        for item in self._items:
            item.assign(up)

    def generate(self, spikes_per_cluster):
        """Populate the cache for all registered fields and the specified
        clusters."""
        assert isinstance(spikes_per_cluster, dict)
        self._spikes_per_cluster = spikes_per_cluster
        # self._store.delete(clusters)
        if hasattr(self._model, 'name'):
            name = self._model.name
        else:
            name = 'the current model'
        debug("Generating the cluster store for {0:s}...".format(name))
        for item in self._items:
            item.store_all_clusters(spikes_per_cluster)
        debug("Done!")


class StoreItem(object):
    """A class describing information stored in the cluster store.

    Attributes
    ----------
    fields : list
        A list of pairs (field_name, storage_location).
        storage_location is either 'memory', 'disk'.
    model : Model
        A Model instance for the current dataset.
    store : ClusterStore
        The ClusterStore instance for the current dataset.

    Methods
    -------
    store_cluster(cluster, spikes)
        Extract some data from the model and store it in the cluster store.
    assign(up)
        Update the store when the clustering changes.
    merge(up)
        Update the store when a merge happens (by default, it is just
        an assign, but this method may be overriden for performance reasons).

    """
    fields = None  # list of (field_name, storage_location)
    name = 'item'

    def __init__(self, model=None, memory_store=None, disk_store=None):
        self.model = model
        self.memory_store = memory_store
        self.disk_store = disk_store

    def store_all_clusters(self, spikes_per_cluster):
        """Copy all data for that item from the model to the cluster store."""
        clusters = sorted(spikes_per_cluster.keys())
        for cluster in clusters:
            debug("Loading {0:s}, cluster {1:d}...".format(self.name,
                  cluster))
            self.store_cluster(cluster, spikes_per_cluster[cluster])

    def merge(self, up):
        """May be overridden."""
        self.assign(up)

    def assign(self, up):
        """May be overridden. No need to delete old clusters here."""
        for cluster in up.added:
            self.store_cluster(cluster, up.new_spikes_per_cluster[cluster])

    def store_cluster(self, cluster, spikes):
        """May be overridden. No need to delete old clusters here."""
        pass
