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

    def keys(self):
        """List of cluster ids in the store."""
        return sorted(self._ds.keys())

    def delete(self, clusters):
        """Delete some clusters from the store."""
        assert isinstance(clusters, list)
        for cluster in clusters:
            if cluster in self._ds:
                del self._ds[cluster]


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
        # if mode == 'r' and not self._cluster_file_exists(cluster):
        #     raise IOError("The cluster file does not exist.")
        return open_h5(path, mode)

    # Data get/set methods
    # -------------------------------------------------------------------------

    def _get(self, f, key):
        """Return the data for a given key.

        Can be overriden for custom on-disk format.

        """
        try:
            return f.read('/{0:s}'.format(key))[...]
        except IOError:
            return None

    def _set(self, f, key, value):
        """Set the data for a given key.

        Can be overriden for custom on-disk format.

        """
        # debug("Writing", key, str(value))
        f.write('/{0:s}'.format(key), value, overwrite=True)

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

    def keys(self):
        """List of cluster ids in the store."""
        files = os.listdir(self._directory)
        clusters = [int(op.splitext(file)[0]) for file in files]
        return sorted(clusters)

    def delete(self, clusters):
        """Delete some clusters from the store."""
        for cluster in clusters:
            if self._cluster_file_exists(cluster):
                os.remove(self._cluster_path(cluster))
