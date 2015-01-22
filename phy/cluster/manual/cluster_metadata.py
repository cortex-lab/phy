# -*- coding: utf-8 -*-

"""Cluster metadata structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict, OrderedDict
from copy import deepcopy

from ...ext.six import iterkeys, itervalues
from ._utils import _unique, _spikes_in_clusters
from ._history import History


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

    def __init__(self, fields=None, data=None):
        if fields is None:
            fields = DEFAULT_FIELDS
        # 'fields' is a list of tuples (field_name, default_value).
        # 'self._fields' is an OrderedDict {field_name ==> default_value}.
        self._fields = OrderedDict(fields)
        self._field_names = list(iterkeys(self._fields))
        # '_data' maps cluster labels to dict (field => value).
        if data is None:
            data = {}
        self._data = defaultdict(dict, data)
        self._spike_clusters = None
        self._cluster_labels = None
        # The stack contains (clusters, field, value) tuples.
        self._undo_stack = History()
        # Keep a deep copy of the original structure for the undo stack.
        self._data_base = deepcopy(self._data)

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

    def _add_clusters(self, clusters):
        """Add new clusters in the structure."""
        for cluster in clusters:
            self._data[cluster] = OrderedDict()

    def _delete_clusters(self, clusters):
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
        self._add_clusters(to_add)
        # Delete the empty clusters.
        to_delete = data_keys - clusters
        self._delete_clusters(to_delete)

    def _set_one(self, cluster, field, value):
        self._data[cluster][field] = value

    def _set_multi(self, clusters, field, values):
        if hasattr(values, '__len__'):
            assert len(clusters) == len(values)
            for cluster, value in zip(clusters, values):
                self._set_one(cluster, field, value)
        else:
            for cluster in clusters:
                self._set_one(cluster, field, values)

    def set(self, clusters, field, values):
        """Set some information for a number of clusters."""
        self._set_multi(clusters, field, values)
        self._undo_stack.add((clusters, field, values))

    def __setitem__(self, item, value):
        clusters, field = item
        self.set(clusters, field, value)

    def _get_one(self, cluster, field):
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

    def _get_multi(self, clusters, field):
        return OrderedDict((cluster, self._get_one(cluster, field))
                           for cluster in clusters)

    def get(self, key, field=None):
        """Get information about one or several clusters."""
        if key in self._field_names:
            # 'key' is a field name; return that field for all clusters.
            assert field is None
            return self._get_multi(self._cluster_labels, key)
        else:
            if hasattr(key, '__len__'):
                # 'key' is a list of clusters, and 'field' must be specified.
                assert field is not None
                return self._get_multi(key, field)
            # 'key' is one or several clusters: return the given field
            elif key in self._data:
                # 'key' is a cluster.
                if field is not None:
                    # 'field' is specified: return just one value.
                    return self._get_one(key, field)
                else:
                    # 'field' is unspecified: return all fields.
                    return OrderedDict((field, self._get_one(key, field))
                                       for field in self._field_names)
        raise ValueError("Key {0:s} not in the list ".format(str(key)) +
                         "of fields {0:s}".format(str(self._field_names)))

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            key, field = key
        else:
            field = None
        return self.get(key, field)

    def undo(self):
        """Undo the last metadata change."""
        self._undo_stack.back()
        self._data = deepcopy(self._data_base)
        for clusters, field, values in self._undo_stack:
            self._set_multi(clusters, field, values)

    def redo(self):
        """Redo the next metadata change."""
        args = self._undo_stack.forward()
        if args is not None:
            self.set(*args)
