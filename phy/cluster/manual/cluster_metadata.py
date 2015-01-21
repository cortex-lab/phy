# -*- coding: utf-8 -*-

"""Cluster metadata structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import namedtuple, defaultdict, OrderedDict

import numpy as np

from ...ext.six import iterkeys, itervalues
from ._utils import _unique, _spikes_in_clusters


#------------------------------------------------------------------------------
# Global variables related to cluster metadata
#------------------------------------------------------------------------------

DEFAULT_GROUPS = [
    (0, 'Noise'),
    (1, 'MUA'),
    (2, 'Good'),
    (3, 'Unsorted'),
]


DEFAULT_FIELDS = [
    ('group', 3),
    ('color', 1),  # TODO: random_color function
]


#------------------------------------------------------------------------------
# ClusterMetadata class
#------------------------------------------------------------------------------

class ClusterMetadata(object):
    """Object holding cluster metadata.

    Constructor
    -----------

    fields : list
        List of tuples (field_name, default_value).

    """

    def __init__(self, fields=None):
        if fields is None:
            fields = DEFAULT_FIELDS
        # 'fields' is a list of tuples (field_name, default_value).
        # 'self._fields' is an OrderedDict {field_name ==> default_value}.
        self._fields = OrderedDict(fields)
        self._field_names = list(iterkeys(self._fields))
        # '_data' maps cluster labels to dict (field => value).
        self._data = defaultdict(dict)
        self._spike_clusters = None
        self._cluster_labels = None

    @property
    def data(self):
        """Dictionary holding data for all clusters."""
        return self._data

    @property
    def spike_clusters(self):
        """Mapping spike ==> cluster label."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        self._spike_clusters = value
        self.update()

    @property
    def cluster_labels(self):
        """Sorted list of non-empty cluster labels."""
        return self._cluster_labels

    def add_clusters(self, clusters):
        """Add new clusters in the structure."""
        for cluster in clusters:
            self._data[cluster] = OrderedDict()

    def delete_clusters(self, clusters):
        """Delete clusters from the structure."""
        for cluster in clusters:
            del self._data[cluster]

    def update(self, cluster_labels=None):
        """Remove empty clusters in the structure.

        Parameters
        ----------

        cluster_labels : array_like
            Sorted list of all cluster labels. By default, the list of unique
            clusters appearing in `self.spike_clusters`.

        """
        if cluster_labels is None:
            cluster_labels = _unique(self._spike_clusters)
        # Update the list of unique cluster labels.
        self._cluster_labels = cluster_labels
        # Find the clusters to add and remove in the structure.
        data_keys = set(iterkeys(self._data))
        clusters = set(self._cluster_labels)
        # Add the new clusters.
        to_add = clusters - data_keys
        self.add_clusters(to_add)
        # Delete the empty clusters.
        to_delete = data_keys - clusters
        self.delete_clusters(to_delete)

    def set(self, cluster, field, value):
        """Set some information for a cluster."""
        self._data[cluster][field] = value

    def get(self, cluster, field):
        """Get a specific information for a given cluster."""
        if cluster in self._data:
            info = self._data[cluster]
            if field in info:
                # Return the value.
                return info[field]
            else:
                # Or return the default value.
                default = self._fields[field]
                # Default is a function ==> call it with the cluster label.
                if hasattr(default, '__call__'):
                    return default(cluster)
                else:
                    return default

    def __getitem__(self, key):
        """Return the field values of all clusters, or all information of a
        cluster."""
        if key in self._data:
            # 'key' is a cluster label.
            return OrderedDict((field, self.get(key, field))
                               for field in self._field_names)
        elif key in self._field_names:
            # 'key' is a field name.
            return OrderedDict((cluster, self.get(cluster, key))
                               for cluster in self.cluster_labels)
        else:
            raise ValueError("Key {0:s} not in the list ".format(str(key)) +
                             "of fields {0:s}".format(str(self._field_names)))
