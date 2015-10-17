# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from six import integer_types

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
    save_requested(spike_clusters, cluster_groups)

    """

    default_shortcuts = {
        'save': 'Save',
        # Wizard actions.
        'next': 'space',
        'previous': 'shift+space',
        'reset_wizard': 'ctrl+alt+space',
        'first': 'MoveToStartOfLine',
        'last': 'MoveToEndOfLine',
        'pin': 'return',
        'unpin': 'Back',
        # Clustering actions.
        'merge': 'g',
        'split': 'k',
        'undo': 'Undo',
        'redo': 'Redo',
    }

    def __init__(self, spike_clusters=None,
                 cluster_groups=None,
                 n_spikes_max_per_cluster=100,
                 shortcuts=None,
                 ):

        self.n_spikes_max_per_cluster = n_spikes_max_per_cluster

        # Load default shortcuts, and override any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        if shortcuts:
            self.shortcuts.update(shortcuts)

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

        @self.cluster_meta.connect  # noqa
        def on_cluster(up):
            if up.history:
                logger.info(up.history.title() + " move.")
            else:
                logger.info("Move clusters %s to %s.",
                            ', '.join(map(str, up.metadata_changed)),
                            up.metadata_value)

        @self.wizard.connect
        def on_select(cluster_ids):
            """When the wizard selects clusters, choose a spikes subset
            and emit the `select` event on the GUI."""
            logger.debug("Select clusters %s.",
                         ', '.join(map(str, cluster_ids)))

        _attach_wizard(self.wizard, self.clustering, self.cluster_meta)

        # Create the actions.
        self._create_actions()

    def _add_action(self, callback, name=None, alias=None):
        name = name or callback.__name__
        shortcut = self.shortcuts.get(name, None)
        self.actions.add(callback=callback,
                         name=name,
                         shortcut=shortcut, alias=alias)

    def _create_actions(self):
        self.actions = Actions()

        # Selection.
        self._add_action(self.select, alias='c')

        # Wizard.
        self._add_action(self.wizard.restart, name='reset_wizard')
        self._add_action(self.wizard.previous)
        self._add_action(self.wizard.next)
        self._add_action(self.wizard.next_by_quality)
        self._add_action(self.wizard.next_by_similarity)
        self._add_action(self.wizard.pin)
        self._add_action(self.wizard.unpin)

        # Clustering.
        self._add_action(self.merge)
        self._add_action(self.split)
        self._add_action(self.move)
        self._add_action(self.undo)
        self._add_action(self.redo)

    def attach(self, gui):
        self.gui = gui

        @self.wizard.connect
        def on_select(cluster_ids):
            """When the wizard selects clusters, choose a spikes subset
            and emit the `select` event on the GUI."""
            spike_ids = select_spikes(np.array(cluster_ids),
                                      self.n_spikes_max_per_cluster,
                                      self.clustering.spikes_per_cluster)
            gui.emit('select', cluster_ids, spike_ids)

        @self.wizard.connect
        def on_start():
            gui.emit('wizard_start')

        # Attach the GUI and register the actions.
        self.actions.attach(gui)

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
