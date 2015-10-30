# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from six import string_types

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from .wizard import Wizard
from phy.gui.actions import Actions
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
    assert group is None or isinstance(group, string_types)
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
# Clustering GUI component
# -----------------------------------------------------------------------------

class ManualClustering(object):
    """Component that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Wizard
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
        'next_by_quality': 'space',
        'previous': 'shift+space',
        'reset_wizard': 'ctrl+alt+space',
        'first': 'MoveToStartOfLine',
        'last': 'MoveToEndOfLine',
        'pin': 'return',
        'unpin': 'Backspace',
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

        # Create the wizard and attach it to Clustering/ClusterMeta.
        self.wizard = Wizard()

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

        _attach_wizard(self.wizard, self.clustering, self.cluster_meta)

    def _create_actions(self, gui):
        self.actions = Actions(gui, default_shortcuts=self.shortcuts)

        # Selection.
        self.actions.add(self.select, alias='c')

        # Wizard.
        self.actions.add(self.wizard.restart, name='reset_wizard')
        self.actions.add(self.wizard.previous)
        self.actions.add(self.wizard.next_by_quality)
        self.actions.add(self.wizard.next_by_similarity)
        self.actions.add(self.wizard.next)  # no shortcut
        self.actions.add(self.wizard.pin)
        self.actions.add(self.wizard.unpin)

        # Clustering.
        self.actions.add(self.merge)
        self.actions.add(self.split)
        self.actions.add(self.move)
        self.actions.add(self.undo)
        self.actions.add(self.redo)

    def attach(self, gui):
        self.gui = gui

        @self.wizard.connect
        def on_select(cluster_ids):
            """When the wizard selects clusters, choose a spikes subset
            and emit the `select` event on the GUI."""
            spike_ids = select_spikes(np.array(cluster_ids),
                                      self.n_spikes_max_per_cluster,
                                      self.clustering.spikes_per_cluster)
            logger.debug("Select clusters: %s (%d spikes).",
                         ', '.join(map(str, cluster_ids)), len(spike_ids))

            if self.gui:
                self.gui.emit('select', cluster_ids, spike_ids)

        @self.wizard.connect
        def on_start():
            if self.gui:
                gui.emit('wizard_start')

        # Create the actions.
        self._create_actions(gui)

        return self

    # Wizard-related actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids):
        # HACK: allow for select(1, 2, 3) in addition to select([1, 2, 3])
        # This makes it more convenient to select multiple clusters with
        # the snippet: ":c 1 2 3".
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
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
