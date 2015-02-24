# -*- coding: utf-8 -*-

"""Cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict


#------------------------------------------------------------------------------
# Data stores
#------------------------------------------------------------------------------

class MemoryStore(object):
    """Store cluster-related data in memory."""
    def __init__(self):
        self._ds = {}

    def store(self, cluster, **data):
        """Store cluster-related data."""
        self._ds[cluster].update(data)

    def load(self, cluster, keys=None):
        """Load cluster-related data."""
        if keys is None:
            return self._ds.get(cluster, {})
        else:
            return {key: self._ds[cluster].get(key, None) for key in keys}

    def delete(self, clusters):
        """Delete some clusters from the store."""
        for cluster in clusters:
            if cluster in self._ds:
                del self._ds[cluster]


class DiskStore(object):
    """Store cluster-related data on disk."""
    def __init__(self, directory):
        self._directory = directory
