# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from phy.stats.clusters import (mean,
                                max_waveform_amplitude,
                                mean_masked_features_distance,
                                )
from phy.gui.actions import Actions
from phy.gui.widgets import Table
from phy.io.array import select_spikes

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

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


def default_wizard_functions(waveforms=None,
                             features=None,
                             masks=None,
                             n_features_per_channel=None,
                             spikes_per_cluster=None,
                             ):
    spc = spikes_per_cluster
    nfc = n_features_per_channel

    def max_waveform_amplitude_quality(cluster):
        spike_ids = select_spikes(cluster_ids=[cluster],
                                  max_n_spikes_per_cluster=100,
                                  spikes_per_cluster=spc,
                                  )
        m = np.atleast_2d(masks[spike_ids])
        w = np.atleast_3d(waveforms[spike_ids])
        mean_masks = mean(m)
        mean_waveforms = mean(w)
        q = max_waveform_amplitude(mean_masks, mean_waveforms)
        logger.debug("Computed cluster quality for %d: %.3f.",
                     cluster, q)
        return q

    def mean_masked_features_similarity(c0, c1):
        s0 = select_spikes(cluster_ids=[c0],
                           max_n_spikes_per_cluster=100,
                           spikes_per_cluster=spc,
                           )
        s1 = select_spikes(cluster_ids=[c1],
                           max_n_spikes_per_cluster=100,
                           spikes_per_cluster=spc,
                           )

        f0 = features[s0]
        m0 = np.atleast_2d(masks[s0])

        f1 = features[s1]
        m1 = np.atleast_2d(masks[s1])

        mf0 = mean(f0)
        mm0 = mean(m0)

        mf1 = mean(f1)
        mm1 = mean(m1)

        d = mean_masked_features_distance(mf0, mf1, mm0, mm1,
                                          n_features_per_channel=nfc,
                                          )
        d = 1. / max(1e-10, d)  # From distance to similarity.
        logger.debug("Computed cluster similarity for (%d, %d): %.3f.",
                     c0, c1, d)
        return d

    return (max_waveform_amplitude_quality,
            mean_masked_features_similarity)


# -----------------------------------------------------------------------------
# Clustering GUI component
# -----------------------------------------------------------------------------

class ManualClustering(object):
    """Component that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Parameters
    ----------

    spike_clusters : ndarray
    cluster_groups : dictionary
    n_spikes_max_per_cluster : int
    shortcuts : dict
    quality_func : function
    similarity_func : function

    GUI events
    ----------

    When this component is attached to a GUI, the GUI emits the following
    events:

    select(cluster_ids, spike_ids)
        when clusters are selected
    cluster(up)
        when a merge or split happens
    save_requested(spike_clusters, cluster_groups)
        when a save is requested by the user

    """

    default_shortcuts = {
        'save': 'Save',
        # Wizard actions.
        'next': 'space',
        'previous': 'shift+space',
        'reset_wizard': 'ctrl+alt+space',
        # Clustering actions.
        'merge': 'g',
        'split': 'k',
        'undo': 'Undo',
        'redo': 'Redo',
    }

    def __init__(self,
                 spike_clusters,
                 cluster_groups=None,
                 n_spikes_max_per_cluster=100,
                 shortcuts=None,
                 ):

        self.gui = None
        self.n_spikes_max_per_cluster = n_spikes_max_per_cluster

        # Load default shortcuts, and override any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Create Clustering and ClusterMeta.
        self.clustering = Clustering(spike_clusters)
        self.cluster_meta = create_cluster_meta(cluster_groups)
        self._global_history = GlobalHistory(process_ups=_process_ups)

        # Log the actions.
        @self.clustering.connect
        def on_cluster(up):
            if up.history:
                logger.info(up.history.title() + " cluster assign.")
            elif up.description == 'merge':
                logger.info("Merge clusters %s to %s.",
                            ', '.join(map(str, up.deleted)),
                            up.added[0])
            else:
                logger.info("Assigned %s spikes.", len(up.spike_ids))

            if self.gui:
                self.gui.emit('cluster', up)

        @self.cluster_meta.connect  # noqa
        def on_cluster(up):
            if up.history:
                logger.info(up.history.title() + " move.")
            else:
                logger.info("Move clusters %s to %s.",
                            ', '.join(map(str, up.metadata_changed)),
                            up.metadata_value)

            if self.gui:
                self.gui.emit('cluster', up)

        # Create the cluster views.
        self._create_cluster_views()

    # Internal methods
    # -------------------------------------------------------------------------

    def _create_actions(self, gui):
        self.actions = Actions(gui, default_shortcuts=self.shortcuts)

        # Selection.
        self.actions.add(self.select, alias='c')

        # Clustering.
        self.actions.add(self.merge)
        self.actions.add(self.split)
        self.actions.add(self.move)
        self.actions.add(self.undo)
        self.actions.add(self.redo)

    def _create_cluster_views(self):
        # Create the cluster view.
        self.cluster_view = cluster_view = Table()
        cluster_view.show()

        # Create the similarity view.
        self.similarity_view = similarity_view = Table()
        similarity_view.show()

        # Selection in the cluster view.
        @cluster_view.connect_
        def on_select(cluster_ids):
            # Emit GUI.select when the selection changes in the cluster view.
            self._emit_select(cluster_ids)
            # Pin the clusters and update the similarity view.
            self._update_similarity_view(cluster_ids)

        # Selection in the similarity view.
        @similarity_view.connect_  # noqa
        def on_select(cluster_ids):
            # Select the clusters from both views.
            cluster_ids = cluster_view.selected + cluster_ids
            self._emit_select(cluster_ids)

        # Save the current selection when an action occurs.
        def on_request_undo_state(up):
            return {'selection': (self.cluster_view.selected,
                                  self.similarity_view.selected)}

        self.clustering.connect(on_request_undo_state)
        self.cluster_meta.connect(on_request_undo_state)

    def _update_cluster_view(self):
        assert self.quality_func
        cols = ['id', 'quality']
        items = [{'id': clu,
                  'quality': self.quality_func(clu),
                  'skip': self.cluster_meta.get('group', clu) in
                  ('noise', 'mua'),
                  }
                 for clu in self.clustering.cluster_ids]
        # TODO: custom measures
        self.cluster_view.set_data(items, cols)

    def _emit_select(self, cluster_ids):
        """Choose spikes from the specified clusters and emit the
        `select` event on the GUI."""
        # Choose a spike subset.
        spike_ids = select_spikes(np.array(cluster_ids),
                                  self.n_spikes_max_per_cluster,
                                  self.clustering.spikes_per_cluster)
        logger.debug("Select clusters: %s (%d spikes).",
                     ', '.join(map(str, cluster_ids)), len(spike_ids))
        if self.gui:
            self.gui.emit('select', cluster_ids, spike_ids)

    # Public methods
    # -------------------------------------------------------------------------

    def set_quality_func(self, f):
        self.quality_func = f

        self._update_cluster_view()
        self.cluster_view.sort_by('quality', 'desc')

    def set_similarity_func(self, f):
        self.similarity_func = f

    def attach(self, gui):
        self.gui = gui

        # Create the actions.
        self._create_actions(gui)

        # Add the cluster views.
        gui.add_view(self.cluster_view, title='ClusterView')
        gui.add_view(self.similarity_view, title='SimilarityView')

        # Update the cluster views and selection when a cluster event occurs.
        @self.gui.connect_
        def on_cluster(up):
            # Get the current sort of the cluster view.
            sort = self.cluster_view.current_sort
            sel_1 = self.similarity_view.selected

            if up.added:
                # Reinitialize the cluster view.
                self._update_cluster_view()
                # Reset the previous sort options.
                if sort[0]:
                    self.cluster_view.sort_by(*sort)

            # Select all new clusters in view 1.
            if up.history == 'undo':
                # Select the clusters that were selected before the undone
                # action.
                clusters_0, clusters_1 = up.undo_state[0]['selection']
                self.cluster_view.select(clusters_0)
                self.similarity_view.select(clusters_1)
            elif up.added:
                self.select(up.added)
                if sel_1:
                    self.similarity_view.next()
            elif up.metadata_changed:
                # Select next in similarity view if all moved are in that view.
                if set(up.metadata_changed) <= set(sel_1):
                    self.similarity_view.next()
                # Otherwise, select next in cluster view.
                else:
                    self.cluster_view.next()
            else:
                self.cluster_view.next()
        return self

    # Selection actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids):
        """Select action: select clusters in the cluster view."""
        # HACK: allow for `select(1, 2, 3)` in addition to `select([1, 2, 3])`
        # This makes it more convenient to select multiple clusters with
        # the snippet: `:c 1 2 3` instead of `:c 1,2,3`.
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
        # Update the cluster view selection.
        self.cluster_view.select(cluster_ids)

    def _update_similarity_view(self, cluster_ids=None):
        """Update the similarity view with matches for the specified
        clusters."""
        assert self.similarity_func
        if cluster_ids is None or not len(cluster_ids):
            cluster_ids = self.cluster_view.selected
        # NOTE: we can also implement similarity wrt several clusters
        sel = cluster_ids[0]
        cols = ['id', 'similarity']
        # TODO: skip
        items = [{'id': clu,
                  'similarity': self.similarity_func(sel, clu)}
                 for clu in self.clustering.cluster_ids
                 if clu not in cluster_ids]
        self.similarity_view.set_data(items, cols)
        self.similarity_view.sort_by('similarity', 'desc')

    @property
    def selected(self):
        return self.cluster_view.selected + self.similarity_view.selected

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.selected
        if len(cluster_ids or []) <= 1:
            return
        self.clustering.merge(cluster_ids)
        self._global_history.action(self.clustering)

    def split(self, spike_ids):
        if len(spike_ids) == 0:
            return
        # TODO: connect to request_split emitted by view
        self.clustering.split(spike_ids)
        self._global_history.action(self.clustering)

    def move(self, cluster_ids, group):
        if len(cluster_ids) == 0:
            return
        self.cluster_meta.set('group', cluster_ids, group)
        self._global_history.action(self.cluster_meta)

    def undo(self):
        self._global_history.undo()

    def redo(self):
        self._global_history.redo()

    def save(self):
        spike_clusters = self.clustering.spike_clusters
        groups = {c: self.cluster_meta.get('group', c)
                  for c in self.clustering.cluster_ids}
        self.gui.emit('save_requested', spike_clusters, groups)
