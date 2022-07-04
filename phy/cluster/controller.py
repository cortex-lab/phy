# -*- coding: utf-8 -*-

"""Perform clustering and metadata actions."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from functools import partial, wraps
import inspect
import logging
from pprint import pprint
from typing import Tuple, Callable, Optional

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta, ClusterMeta
from .clustering import Clustering

from phylib.utils import Bunch, emit, connect, unconnect, silent
from phy.gui.actions import Actions
from phy.gui.qt import _block, set_busy, _wait
from phy.gui.widgets import Table, _uniq

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Util functions
# ----------------------------------------------------------------------------

def _process_ups(ups):  # pragma: no cover
    """This function processes the UpdateInfo instances of the two
    undo stacks (clustering and cluster metadata) and concatenates them
    into a single UpdateInfo instance."""
    if len(ups) == 0:
        return
    elif len(ups) == 1:
        return ups[0]
    elif len(ups) == 2:
        up = ups[0]
        up.update(ups[1])
        return up
    else:
        raise NotImplementedError()


def _ensure_all_ints(l):
    if (l is None or l == []):
        return
    for i in range(len(l)):
        l[i] = int(l[i])


# ----------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------

class Controller:
    """Component that brings manual clustering facilities to a GUI:

    * `Clustering` instance: merge, split, undo, redo.
    * `ClusterMeta` instance: change cluster metadata (e.g. group).
    * Cluster selection.
    * Many manual clustering-related actions, snippets, shortcuts, etc.
    * Two HTML tables : `ClusterView` and `SimilarityView`.

    Constructor
    -----------

    spike_clusters : array-like
        Spike-clusters assignments.
    cluster_groups : dict
        Maps a cluster id to a group name (noise, mea, good, None for unsorted).
    cluster_metrics : dict
        Maps a metric name to a function `cluster_id => value`
    similarity : function
        Maps a cluster id to a list of pairs `[(similar_cluster_id, similarity), ...]`
    new_cluster_id : function
        Function that takes no argument and returns a brand new cluster id (smallest cluster id
        not used in the cache).
    sort : 2-tuple
        Initial sort as a pair `(column_name, order)` where `order` is either `asc` or `desc`
    context : Context
        Handles the cache.

    Events
    ------

    When this component is attached to a GUI, the following events are emitted:

    * `select(cluster_ids)`
        When clusters are selected in the cluster view or similarity view.
    * `cluster(up)`
        When a clustering action occurs, changing the spike clusters assignment of the cluster
        metadata.
    * `attach_gui(gui)`
        When the Supervisor instance is attached to the GUI.
    * `request_split()`
        When the user requests to split (typically, a lasso has been drawn before).
    * `save_clustering(spike_clusters, cluster_groups, *cluster_labels)`
        When the user wants to save the spike cluster assignments and the cluster metadata.

    """

    def __init__(
        self,
        spike_clusters=None,
        cluster_groups=None,
        cluster_labels=None,
        cluster_metrics=None,
        similarity=None,
        new_cluster_id=None,
        context=None,
    ):

        self.fn_similarity = similarity  # function cluster => [(cl, sim), ...]
        self.context = context

        # Cluster metrics.
        # This is a dict {name: func cluster_id => value}.
        self.cluster_metrics = cluster_metrics or {}
        self.cluster_metrics['n_spikes'] = self.n_spikes

        # Cluster labels.
        # This is a dict {name: {cl: value}}
        self.cluster_labels = cluster_labels or {}

        # Load the cached spikes_per_cluster array.
        spc = context.load('spikes_per_cluster') if context else None

        # Create Clustering.
        self.clustering = Clustering(
            spike_clusters, spikes_per_cluster=spc, new_cluster_id=new_cluster_id)

        # Cache the spikes_per_cluster array.
        self._save_spikes_per_cluster()

        # Create the ClusterMeta instance.
        self.cluster_meta = create_cluster_meta(cluster_groups or {})

        # Add the labels.
        for label, values in self.cluster_labels.items():
            if label == 'group':
                continue
            self.cluster_meta.add_field(label)
            for cl, v in values.items():
                self.cluster_meta.set(label, [cl], v, add_to_stack=False)

        # Create the GlobalHistory instance.
        self._global_history = GlobalHistory(process_ups=_process_ups)

        # Raise cluster
        @connect(sender=self.clustering)
        def on_cluster(sender, up):
            # NOTE: update the cluster meta of new clusters, depending on the values of the
            # ancestor clusters. In case of a conflict between the values of the old clusters,
            # the largest cluster wins and its value is set to its descendants.
            if up.added:
                self.cluster_meta.set_from_descendants(
                    up.descendants, largest_old_cluster=up.largest_old_cluster)
            emit('cluster', self, up)

        @connect(sender=self.cluster_meta)  # noqa
        def on_cluster(sender, up):  # noqa
            emit('cluster', self, up)

        connect(self._save_new_cluster_id, event='cluster', sender=self)

    # Internal methods
    # -------------------------------------------------------------------------

    def _save_spikes_per_cluster(self):
        """Cache on the disk the dictionary with the spikes belonging to each cluster."""
        if self.context:
            self.context.save('spikes_per_cluster',
                              self.clustering.spikes_per_cluster, kind='pickle')

    def _log_action(self, sender, up):
        """Log the clustering action (merge, split)."""
        if sender != self.clustering:
            return
        if up.history:
            logger.info(up.history.title() + " cluster assign.")
        elif up.description == 'merge':
            logger.info("Merge clusters %s to %s.", ', '.join(map(str, up.deleted)), up.added[0])
        else:
            logger.info("Assigned %s spikes.", len(up.spike_ids))

    def _log_action_meta(self, sender, up):
        """Log the cluster meta action (move, label)."""
        if sender != self.cluster_meta:
            return
        if up.history:
            logger.info(up.history.title() + " move.")
        else:
            logger.info(
                "Change %s for clusters %s to %s.", up.description,
                ', '.join(map(str, up.metadata_changed)), up.metadata_value)

        # Skip cluster metadata other than groups.
        if up.description != 'metadata_group':
            return

    def _save_new_cluster_id(self, sender, up):
        """Save the new cluster id on disk, knowing that cluster ids are unique for
        easier cache consistency."""
        if up.description not in ('assign', 'merge'):
            return
        new_cluster_id = self.clustering.new_cluster_id()
        if self.context:
            logger.log(5, "Save the new cluster id: %d.", new_cluster_id)
            self.context.save('new_cluster_id', dict(new_cluster_id=new_cluster_id))

    # Cluster info
    # -------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """List of cluster ids."""
        return self.clustering.cluster_ids

    def n_spikes(self, cluster_id):
        """Number of spikes in a given cluster."""
        return len(self.clustering.spikes_per_cluster.get(cluster_id, []))

    def cluster_group(self, cluster_id):
        """Return the group of a cluster."""
        return self.cluster_meta.get('group', cluster_id)

    @property
    def fields(self):
        """List of all cluster label names."""
        return tuple(f for f in self.cluster_meta.fields if f not in ('group',))

    def get_labels(self, field):
        """Return the labels of all clusters, for a given label name."""
        return {c: self.cluster_meta.get(field, c) for c in self.clustering.cluster_ids}

    # Clustering action
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids, to=None):
        """Merge the selected clusters."""
        if cluster_ids is None:
            return
        if len(cluster_ids or []) <= 1:
            return
        assert to is not None
        out = self.clustering.merge(cluster_ids, to=to)
        self._global_history.action(self.clustering)
        return out

    def split(self, spike_ids=None, spike_clusters_rel=0):
        """Make a new cluster out of the specified spikes."""
        if spike_ids is None:
            # Concatenate all spike_ids returned by views who respond to request_split.
            spike_ids = emit('request_split', self)
            spike_ids = np.concatenate(spike_ids).astype(np.int64)
            assert spike_ids.dtype == np.int64
            assert spike_ids.ndim == 1
        if len(spike_ids) == 0:
            logger.warning(
                """No spikes selected, cannot split.""")
            return
        out = self.clustering.split(
            spike_ids, spike_clusters_rel=spike_clusters_rel)
        self._global_history.action(self.clustering)
        return out

    def label(self, cluster_ids=None, name=None, value=None):
        """Assign a label to some clusters."""
        if cluster_ids is None:
            return
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        if len(cluster_ids) == 0:
            return
        out = self.cluster_meta.set(name, cluster_ids, value)
        self._global_history.action(self.cluster_meta)
        return out

    def move(self, cluster_ids=None, group=None):
        """Assign a cluster group to some clusters."""
        if not cluster_ids:
            return
        _ensure_all_ints(cluster_ids)
        logger.debug("Move %s to %s.", cluster_ids, group)
        group = 'unsorted' if group is None else group
        return self.label(cluster_ids=cluster_ids, name='group', value=group)

    def undo(self):
        """Undo the last action."""
        self._global_history.undo()

    def redo(self):
        """Undo the last undone action."""
        self._global_history.redo()

    def save(self):
        """Save the manual clustering back to disk.

        This method emits the `save_clustering(spike_clusters, groups, *labels)` event.
        It is up to the caller to react to this event and save the data to disk.

        """
        spike_clusters = self.clustering.spike_clusters
        groups = {
            c: self.cluster_meta.get('group', c) or 'unsorted'
            for c in self.clustering.cluster_ids}
        # List of tuples (field_name, dictionary).
        labels = [
            (field, self.get_labels(field)) for field in self.cluster_meta.fields
            if field not in ('next_cluster')]
        emit('save_clustering', self, spike_clusters, groups, *labels)
        # Cache the spikes_per_cluster array.
        self._save_spikes_per_cluster()
        self._is_dirty = False
