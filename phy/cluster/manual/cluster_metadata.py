# -*- coding: utf-8 -*-

"""Cluster metadata structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict, OrderedDict
from functools import partial
from copy import deepcopy

from ...plot._color import _random_color
from ...ext.six import iterkeys, itervalues, iteritems
from ._utils import _unique, _spikes_in_clusters
from ._update_info import UpdateInfo
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
    ('color', _random_color),
]


#------------------------------------------------------------------------------
# ClusterMetadata class
#------------------------------------------------------------------------------

def _default_value(field, default):
    """Return the default value of a field."""
    if hasattr(default, '__call__'):
        return default()
    else:
        return default


def _default_info(fields):
    """Default structure holding info of a cluster."""
    return dict([(field, _default_value(field, default))
                 for field, default in fields])


def _cluster_info(fields, data=None):
    """Initialize a structure holding cluster metadata."""
    if data is None:
        data = {}
    out = defaultdict(partial(_default_info, fields))
    for cluster, values in iteritems(data):
        # Create the default cluster info dict.
        info = out[cluster]
        # Update the specified values, so that the default values are used
        # for the unspecified values.
        for key, value in iteritems(values):
            info[key] = value
    return out


class ClusterMetadata(object):
    """Object holding cluster metadata.

    Constructor
    -----------

    fields : list
        List of tuples (field_name, default_value).
    data : dict-like
        Initial data.

    """

    def __init__(self, data=None, fields=None):
        if fields is None:
            fields = DEFAULT_FIELDS
        # 'fields' is a list of tuples (field_name, default_value).
        # 'self._fields' is an OrderedDict {field_name ==> default_value}.
        self._fields = OrderedDict(fields)
        self._field_names = list(iterkeys(self._fields))
        # '_data' maps cluster labels to dict (field => value).
        self._data = _cluster_info(fields, data=data)
        self._spike_clusters = None
        self._cluster_labels = None
        # The stack contains (clusters, field, value, update_info) tuples.
        self._undo_stack = History((None, None, None, None))
        # Keep a deep copy of the original structure for the undo stack.
        self._data_base = deepcopy(self._data)

    @property
    def data(self):
        """Dictionary holding data for all clusters."""
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, field):
        return self._data[key][field]

    def _set_one(self, cluster, field, value):
        """Set information for a given cluster."""
        self._data[cluster][field] = value

    def _set_multi(self, clusters, field, values):
        """Set some information for a number of clusters."""
        if hasattr(values, '__len__'):
            assert len(clusters) == len(values)
            for cluster, value in zip(clusters, values):
                self._set_one(cluster, field, value)
        else:
            for cluster in clusters:
                self._set_one(cluster, field, values)

    def set(self, clusters, field, values):
        """Set some information for a number of clusters and add the changes
        to the undo stack."""
        # Ensure 'clusters' is a list of clusters.
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        self._set_multi(clusters, field, values)
        info = UpdateInfo(description=field, metadata_changed=clusters)
        self._undo_stack.add((clusters, field, values, info))
        return info

    def update(self, up=None):
        """Update cluster metadata after a clustering action."""
        # TODO: what happens to color/group of new clusters after merge/split?
        pass

    def undo(self):
        """Undo the last metadata change."""
        args = self._undo_stack.back()
        if args is None:
            return
        self._data = deepcopy(self._data_base)
        for clusters, field, values, _ in self._undo_stack:
            if clusters is not None:
                self._set_multi(clusters, field, values)
        # Return the UpdateInfo instance of the undo action.
        info = args[-1]
        return info

    def redo(self):
        """Redo the next metadata change."""
        args = self._undo_stack.forward()
        if args is None:
            return
        clusters, field, values, info = args
        self._set_multi(clusters, field, values)
        # Return the UpdateInfo instance of the redo action.
        return info
