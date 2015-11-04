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


def _make_wizard_default_functions(waveforms=None,
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

        logger.debug("Computed cluster similarity for (%d, %d): %.3f.",
                     c0, c1, d)
        return -d  # NOTE: convert distance to score

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

    Bring the `select` event to the GUI. This is raised when clusters are
    selected by the user or by the wizard.

    Parameters
    ----------

    gui : GUI
    spike_clusters : ndarray
    cluster_groups : dictionary
    n_spikes_max_per_cluster : int

    Events
    ------

    select(cluster_ids, spike_ids)
        when clusters are selected
    on_cluster(up)
        when a merge or split happens
    wizard_start()
        when the wizard (re)starts
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
                 quality_func=None,
                 similarity_func=None,
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

        # Wizard functions.
        self.quality_func = quality_func or (lambda c: 0)
        self.similarity_func = similarity_func or (lambda c, d: 0)

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
                # TODO: how many spikes?
                logger.info("Assigned spikes.")

            if self.gui:
                self.gui.emit('on_cluster', up)

        @self.cluster_meta.connect  # noqa
        def on_cluster(up):
            if up.history:
                logger.info(up.history.title() + " move.")
            else:
                logger.info("Move clusters %s to %s.",
                            ', '.join(map(str, up.metadata_changed)),
                            up.metadata_value)

            if self.gui:
                self.gui.emit('on_cluster', up)

        # _attach_wizard(self.wizard, self.clustering, self.cluster_meta)

    def _create_actions(self, gui):
        self.actions = Actions(gui, default_shortcuts=self.shortcuts)

        # Selection.
        self.actions.add(self.select, alias='c')

        # Wizard.
        # self.actions.add(self.wizard.restart, name='reset_wizard')
        # self.actions.add(self.wizard.previous)
        # self.actions.add(self.wizard.next_by_quality)
        # self.actions.add(self.wizard.next_by_similarity)
        # self.actions.add(self.wizard.next)  # no shortcut
        # self.actions.add(self.wizard.pin)
        # self.actions.add(self.wizard.unpin)

        # Clustering.
        self.actions.add(self.merge)
        self.actions.add(self.split)
        self.actions.add(self.move)
        self.actions.add(self.undo)
        self.actions.add(self.redo)

    def _create_cluster_view(self):
        table = Table()
        cols = ['id', 'quality']
        # TODO: skip
        items = [{'id': int(clu), 'quality': self.quality_func(clu)}
                 for clu in self.clustering.cluster_ids]
        table.set_data(items, cols)
        table.build()
        return table

    def _create_similarity_view(self):
        table = Table()
        table.build()
        return table

    def attach(self, gui):
        self.gui = gui

        # Cluster view.
        self.cluster_view = self._create_cluster_view()
        gui.add_view(self.cluster_view, title='ClusterView')

        # Similarity view.
        self.similarity_view = self._create_similarity_view()
        gui.add_view(self.similarity_view, title='SimilarityView')

        def _update_similarity_view(cluster_ids):
            if len(cluster_ids) == 1:
                sel = int(cluster_ids[0])
                cols = ['id', 'similarity']
                # TODO: skip
                items = [{'id': int(clu),
                          'similarity': self.similarity_func(sel, clu)}
                         for clu in self.clustering.cluster_ids]
                self.similarity_view.set_data(items, cols)
                self.similarity_view.sort_by('similarity')
                self.similarity_view.sort_by('similarity')

        def _select(cluster_ids):
            spike_ids = select_spikes(np.array(cluster_ids),
                                      self.n_spikes_max_per_cluster,
                                      self.clustering.spikes_per_cluster)
            logger.debug("Select clusters: %s (%d spikes).",
                         ', '.join(map(str, cluster_ids)), len(spike_ids))

            if self.gui:
                self.gui.emit('select', cluster_ids, spike_ids)

        def on_select1(cluster_ids):
            # Update the similarity view when the selection changes in
            # the cluster view.
            _update_similarity_view(cluster_ids)
            _select(cluster_ids)
        self.cluster_view.connect_(on_select1, event='select')

        def on_select2(cluster_ids):
            # TODO: prepend the clusters selected in the cluster view
            _select(cluster_ids)
        self.similarity_view.connect_(on_select2, event='select')  # noqa

        # Create the actions.
        self._create_actions(gui)

        return self

    # Selection actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids):
        # HACK: allow for select(1, 2, 3) in addition to select([1, 2, 3])
        # This makes it more convenient to select multiple clusters with
        # the snippet: ":c 1 2 3".
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
        # self.wizard.select(cluster_ids)

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None):
        # if cluster_ids is None:
        #     cluster_ids = self.wizard.selection
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
