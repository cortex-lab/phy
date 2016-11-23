# -*- coding: utf-8 -*-

"""Clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from copy import deepcopy
from collections import defaultdict
import logging

from ._history import History
from phy.utils import Bunch, _as_list, _is_list, EventEmitter

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _update_cluster_selection(clusters, up):
    clusters = list(clusters)
    # Remove deleted clusters.
    clusters = [clu for clu in clusters if clu not in up.deleted]
    # Add new clusters at the end of the selection.
    return clusters + [clu for clu in up.added if clu not in clusters]


def _join(clusters):
    return '[{}]'.format(', '.join(map(str, clusters)))


def create_cluster_meta(cluster_groups):
    """Return a ClusterMeta instance with cluster group support."""
    meta = ClusterMeta()
    meta.add_field('group')

    cluster_groups = cluster_groups or {}
    data = {c: {'group': v} for c, v in cluster_groups.items()}
    meta.from_dict(data)

    return meta


#------------------------------------------------------------------------------
# UpdateInfo class
#------------------------------------------------------------------------------

class UpdateInfo(Bunch):
    """Hold information about clustering changes."""
    def __init__(self, **kwargs):
        d = dict(
            description='',  # information about the update: 'merge', 'assign',
                             # or 'metadata_<name>'
            history=None,  # None, 'undo', or 'redo'
            spike_ids=[],  # all spikes affected by the update
            added=[],  # new clusters
            deleted=[],  # deleted clusters
            descendants=[],  # pairs of (old_cluster, new_cluster)
            metadata_changed=[],  # clusters with changed metadata
            metadata_value=None,  # new metadata value
            undo_state=None,  # returned during an undo: it contains
                              # information about the undone action
        )
        d.update(kwargs)
        super(UpdateInfo, self).__init__(d)

    def __repr__(self):
        desc = self.description
        h = ' ({})'.format(self.history) if self.history else ''
        if not desc:
            return '<UpdateInfo>'
        elif desc in ('merge', 'assign'):
            a, d = _join(self.added), _join(self.deleted)
            return '<{desc}{h} {d} => {a}>'.format(desc=desc,
                                                   a=a,
                                                   d=d,
                                                   h=h,
                                                   )
        elif desc.startswith('metadata'):
            c = _join(self.metadata_changed)
            m = self.metadata_value
            return '<{desc}{h} {c} => {m}>'.format(desc=desc,
                                                   c=c,
                                                   m=m,
                                                   h=h,
                                                   )
        return '<UpdateInfo>'


#------------------------------------------------------------------------------
# ClusterMetadataUpdater class
#------------------------------------------------------------------------------

class ClusterMeta(EventEmitter):
    """Handle cluster metadata changes."""
    def __init__(self):
        super(ClusterMeta, self).__init__()
        self._fields = {}
        self._reset_data()

    def _reset_data(self):
        self._data = {}
        self._data_base = {}
        # The stack contains (clusters, field, value, update_info, undo_state)
        # tuples.
        self._undo_stack = History((None, None, None, None, None))

    @property
    def fields(self):
        """List of fields."""
        return sorted(self._fields.keys())

    def add_field(self, name, default_value=None):
        """Add a field with an optional default value."""
        self._fields[name] = default_value

        def func(cluster):
            return self.get(name, cluster)

        setattr(self, name, func)

    def from_dict(self, dic):
        """Import data from a {cluster: {field: value}} dictionary."""
        self._reset_data()
        for cluster, vals in dic.items():
            for field, value in vals.items():
                self.set(field, [cluster], value, add_to_stack=False)
        self._data_base = deepcopy(self._data)

    def to_dict(self, field):
        """Export data to a {cluster: value} dictionary, for a particular
        field."""
        assert field in self._fields, "This field doesn't exist"
        return {cluster: self.get(field, cluster)
                for cluster in self._data.keys()}

    def set(self, field, clusters, value, add_to_stack=True):
        """Set the value of one of several clusters."""
        # Add the field if it doesn't exist.
        if field not in self._fields:
            self.add_field(field)
        assert field in self._fields

        clusters = _as_list(clusters)
        for cluster in clusters:
            if cluster not in self._data:
                self._data[cluster] = {}
            self._data[cluster][field] = value

        up = UpdateInfo(description='metadata_' + field,
                        metadata_changed=clusters,
                        metadata_value=value,
                        )
        undo_state = self.emit('request_undo_state', up)

        if add_to_stack:
            self._undo_stack.add((clusters, field, value, up, undo_state))
            self.emit('cluster', up)

        return up

    def get(self, field, cluster):
        """Retrieve the value of one cluster."""
        if _is_list(cluster):
            return [self.get(field, c) for c in cluster]
        assert field in self._fields
        default = self._fields[field]
        return self._data.get(cluster, {}).get(field, default)

    def set_from_descendants(self, descendants):
        """Update metadata of some clusters given the metadata of their
        ascendants."""
        for field in self.fields:

            # This gives a set of metadata values of all the parents
            # of any new cluster.
            candidates = defaultdict(set)
            for old, new in descendants:
                candidates[new].add(self.get(field, old))

            # Loop over all new clusters.
            for new, vals in candidates.items():
                vals = list(vals)
                default = self._fields[field]
                # If all the parents have the same value, assign it to
                # the new cluster if it is not the default.
                if len(vals) == 1 and vals[0] != default:
                    self.set(field, new, vals[0])
                # Otherwise, the default is assumed.

    def undo(self):
        """Undo the last metadata change.

        Returns
        -------

        up : UpdateInfo instance

        """
        args = self._undo_stack.back()
        if args is None:
            return
        self._data = deepcopy(self._data_base)
        for clusters, field, value, up, undo_state in self._undo_stack:
            if clusters is not None:
                self.set(field, clusters, value, add_to_stack=False)

        # Return the UpdateInfo instance of the undo action.
        up, undo_state = args[-2:]
        up.history = 'undo'
        up.undo_state = undo_state

        self.emit('cluster', up)
        return up

    def redo(self):
        """Redo the next metadata change.

        Returns
        -------

        up : UpdateInfo instance
        """
        args = self._undo_stack.forward()
        if args is None:
            return
        clusters, field, value, up, undo_state = args
        self.set(field, clusters, value, add_to_stack=False)

        # Return the UpdateInfo instance of the redo action.
        up.history = 'redo'

        self.emit('cluster', up)
        return up
