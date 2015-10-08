# -*- coding: utf-8 -*-

"""Manual clustering GUI plugins."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from .wizard import Wizard
from phy.gui.actions import Actions, Snippets
from phy.io.array import select_spikes
from phy.utils.plugin import IPlugin

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


# -----------------------------------------------------------------------------
# Attach wizard to effectors (clustering and cluster_meta)
# -----------------------------------------------------------------------------

_wizard_group_mapping = {
    'noise': 'ignored',
    'mua': 'ignored',
    'good': 'good',
}


def _wizard_group(group):
    # The group should be None, 'mua', 'noise', or 'good'.
    group = group.lower() if group else group
    return _wizard_group_mapping.get(group, None)


def _attach_wizard_to_effector(wizard, effector):

    # Save the current selection when an action occurs.
    @effector.connect
    def on_request_undo_state(up):
        return {'selection': wizard._selection}

    @effector.connect
    def on_cluster(up):
        if not up.history:
            # Reset the history after every change.
            # That's because the history contains references to dead clusters.
            wizard.reset()
        if up.history == 'undo':
            # Revert to the given selection after an undo.
            wizard.select(up.undo_state[0]['selection'], add_to_history=False)


def _attach_wizard_to_clustering(wizard, clustering):
    _attach_wizard_to_effector(wizard, clustering)

    @wizard.set_cluster_ids_function
    def get_cluster_ids():
        return clustering.cluster_ids

    @clustering.connect
    def on_cluster(up):
        if up.added and up.history != 'undo':
            wizard.select([up.added[0]])
            # NOTE: after a merge, select the merged one AND the most similar.
            # There is an ambiguity after a merge: does the merge occurs during
            # a wizard session, in which case we want to pin the merged
            # cluster? If it is just a "cold" merge, then we might not want
            # to pin the merged cluster. But cold merges are supposed to be
            # less frequent than wizard merges.
            wizard.pin()


def _attach_wizard_to_cluster_meta(wizard, cluster_meta):
    _attach_wizard_to_effector(wizard, cluster_meta)

    @wizard.set_status_function
    def status(cluster):
        group = cluster_meta.get('group', cluster)
        return _wizard_group(group)

    @cluster_meta.connect
    def on_cluster(up):
        if up.description == 'metadata_group' and up.history != 'undo':
            cluster = up.metadata_changed[0]
            wizard.next_selection([cluster], ignore_group=True)
            # TODO: pin after a move? Yes if the previous selection >= 2, no
            # otherwise. See similar note above.
            # wizard.pin()


def _attach_wizard(wizard, clustering, cluster_meta):
    _attach_wizard_to_clustering(wizard, clustering)
    _attach_wizard_to_cluster_meta(wizard, cluster_meta)


# -----------------------------------------------------------------------------
# Clustering GUI plugins
# -----------------------------------------------------------------------------

class ManualClustering(IPlugin):
    """Plugin that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Wizard
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Bring the `select` event to the GUI. This is raised when clusters are
    selected by the user or by the wizard.

    Other plugins can connect to that event.

    Parameters
    ----------

    gui : GUI
    spike_clusters : ndarray
    cluster_groups : dictionary
    n_spikes_max_per_cluster : int

    Events
    ------

    select(cluster_ids, spike_ids)
    save_requested(spike_clusters, cluster_groups)

    """
    def attach_to_gui(self, gui,
                      spike_clusters=None,
                      cluster_groups=None,
                      n_spikes_max_per_cluster=100,
                      ):
        self.gui = gui

        # Create Clustering and ClusterMeta.
        self.clustering = Clustering(spike_clusters)
        self.cluster_meta = create_cluster_meta(cluster_groups)
        self._global_history = GlobalHistory(process_ups=_process_ups)

        # Create the wizard and attach it to Clustering/ClusterMeta.
        self.wizard = Wizard()
        _attach_wizard(self.wizard, self.clustering, self.cluster_meta)

        @self.wizard.connect
        def on_select(cluster_ids):
            """When the wizard selects clusters, choose a spikes subset
            and emit the `select` event on the GUI.

            The wizard is responsible for the notion of "selected clusters".

            """
            spike_ids = select_spikes(np.array(cluster_ids),
                                      n_spikes_max_per_cluster,
                                      self.clustering.spikes_per_cluster)
            gui.emit('select', cluster_ids, spike_ids)

        self.create_actions(gui)

        return self

    def create_actions(self, gui):
        self.actions = actions = Actions()
        self.snippets = snippets = Snippets()

        # Create the default actions for the clustering GUI.
        @actions.connect
        def on_reset():
            # Selection.
            actions.add(callback=self.select, alias='c')

            # Wizard.
            actions.add(callback=self.wizard.restart, name='reset_wizard')
            actions.add(callback=self.wizard.previous)
            actions.add(callback=self.wizard.next)
            actions.add(callback=self.wizard.next_by_quality)
            actions.add(callback=self.wizard.next_by_similarity)
            actions.add(callback=self.wizard.pin)
            actions.add(callback=self.wizard.unpin)

            # Clustering.
            actions.add(callback=self.merge)
            actions.add(callback=self.split)
            actions.add(callback=self.move)
            actions.add(callback=self.undo)
            actions.add(callback=self.redo)

        # Attach the GUI and register the actions.
        snippets.attach(gui, actions)
        actions.attach(gui)
        actions.reset()

    # Wizard-related actions
    # -------------------------------------------------------------------------

    def select(self, cluster_ids):
        self.wizard.select(cluster_ids)

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.wizard.selection
        if len(cluster_ids) <= 1:
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
