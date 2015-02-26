# -*- coding: utf-8 -*-

"""Cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import shutil
from collections import defaultdict

from ...utils.logging import debug
from ...utils._misc import (_phy_user_dir,
                            _ensure_phy_user_dir_exists)
from ...io.h5 import open_h5
from ...io.sparse import load_h5, save_h5
from ...ext.six import string_types
from ...ext.slugify import slugify


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

    def _cluster_file(self, cluster, mode):
        """Return a file handle of a cluster file."""
        path = self._cluster_path(cluster)
        return open_h5(path, mode)

    # Data get/set methods
    # -------------------------------------------------------------------------

    def _get(self, f, key):
        """Return the data for a given key."""
        path = '/{0:s}'.format(key)
        return load_h5(f, path)

    def _set(self, f, key, value):
        """Set the data for a given key."""
        path = '/{0:s}'.format(key)
        save_h5(f, path, value, overwrite=True)

    # Public methods
    # -------------------------------------------------------------------------

    def store(self, cluster, **data):
        """Store cluster-related data."""
        with self._cluster_file(cluster, 'a') as f:
            for key, value in data.items():
                self._set(f, key, value)

    def load(self, cluster, keys=None):
        """Load cluster-related data."""
        # The cluster doesn't exist: return None for all keys.
        if not self._cluster_file_exists(cluster):
            if keys is None:
                return {}
            else:
                return {key: None for key in keys}
        # Create the output dictionary.
        out = {}
        # Open the cluster file in read mode.
        with self._cluster_file(cluster, 'r') as f:
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

def _ensure_disk_store_exists(dir_name, root_path=None):
    # Disk store.
    if root_path is None:
        _ensure_phy_user_dir_exists()
        root_path = _phy_user_dir('cluster_store')
    # Create the disk store if it does not exist.
    if not op.exists(root_path):
        os.mkdir(root_path)
    # Put the store in a subfolder, using the name.
    dir_name = slugify(dir_name)
    path = op.join(root_path, dir_name)
    if not op.exists(path):
        os.mkdir(path)
    return path


def _concatenate(*dicts):
    """Concatenate dictionaries."""
    out = {}
    for dic in dicts:
        out.update(dic)
    return out


class BaseClusterStore(object):
    """Hold cluster-related information in memory and on disk."""

    def __init__(self, dir_name, root_path=None):

        # Create the memory store.
        self._memory_store = MemoryStore()

        # Create the disk store.
        path = _ensure_disk_store_exists(dir_name, root_path=root_path)
        self._disk_store = DiskStore(path)

        # Where the info are stored: a {'field' => ('memory' or 'disk')} dict.
        self._dispatch = {}

    def register_field(self, name, location):
        """Register a field to be stored either in 'memory' or on 'disk'."""
        self._check_location(location)
        self._dispatch[name] = location

    def _check_location(self, location):
        """Check that a location is valid."""
        if location not in ('memory', 'disk'):
            raise ValueError("'location 'should be 'memory' or 'disk'.")

    def _filter(self, keys, location):
        """Return all keys registered in the specified location."""
        if keys is None:
            return None
        else:
            return [key for key in keys
                    if self._dispatch.get(key, None) == location]

    # Public methods
    # -------------------------------------------------------------------------

    @property
    def clusters(self):
        """Return the list of clusters present in the store."""
        clusters_memory = self._memory_store.clusters
        clusters_disk = self._disk_store.clusters
        # Both stores should have the same clusters at all times.
        if clusters_memory != clusters_disk:
            raise RuntimeError("Cluster store inconsistency.")
        return clusters_memory

    def store(self, cluster, location=None, **data):
        """Store cluster-related information."""

        # If the location is specified, register the fields there.
        if location in ('memory', 'disk'):
            for key in data.keys():
                self.register_field(key, location)
        elif location is not None:
            self._check_location(location)

        # Store data in memory.
        data_memory = {k: data[k] for k in self._filter(data.keys(), 'memory')}
        self._memory_store.store(cluster, **data_memory)

        # Store data on disk.
        data_disk = {k: data[k] for k in self._filter(data.keys(), 'disk')}
        self._disk_store.store(cluster, **data_disk)

    def load(self, cluster, keys=None):
        """Load cluster-related information."""
        if isinstance(keys, string_types):
            if self._dispatch[keys] == 'memory':
                return self._memory_store.load(cluster, keys)
            elif self._dispatch[keys] == 'disk':
                return self._disk_store.load(cluster, keys)
        elif keys is None or isinstance(keys, list):
            data_memory = self._memory_store.load(cluster,
                                                  self._filter(keys, 'memory'))
            data_disk = self._disk_store.load(cluster,
                                              self._filter(keys, 'disk'))
            return _concatenate(data_memory, data_disk)
        else:
            raise ValueError("'keys' should be a list or a string.")

    def clear(self):
        """Clear the cluster store."""
        self._memory_store.clear()
        self._disk_store.clear()

    def delete(self, clusters):
        """Delete all information about the specified clusters."""
        self._memory_store.delete(clusters)
        self._disk_store.delete(clusters)
