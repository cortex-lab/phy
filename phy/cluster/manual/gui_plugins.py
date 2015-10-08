# -*- coding: utf-8 -*-

"""Manual clustering GUI plugins."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from .wizard import Wizard, _wizard_group
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

def _attach_wizard_to_effector(wizard, effector):

    # Save the current selection when an action occurs.
    @effector.connect
    def on_request_undo_state(up):
        return {'selection': wizard._selection}

    @effector.connect
    def on_cluster(up):
        if not up.history:
            # Reset the history after every change.
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
            wizard.select((up.added[0],))
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
            wizard.select((cluster,))
            wizard.pin()


def _attach_wizard(wizard, clustering, cluster_meta):
    @clustering.connect
    def on_cluster(up):
        # Set the cluster metadata of new clusters.
        if up.added:
            cluster_meta.set_from_descendants(up.descendants)


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
            spike_ids = select_spikes(cluster_ids,
                                      n_spikes_max_per_cluster,
                                      self.clustering.spikes_per_cluster)
            gui.emit('select', cluster_ids, spike_ids)

        self.create_actions(gui)

        return self

    @property
    def cluster_ids(self):
        return self.clustering.cluster_ids

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
        self.clustering.merge(cluster_ids)
        self._global_history.action(self.clustering)

    def split(self, spike_ids):
        # TODO: connect to request_split emitted by view
        self.clustering.split(spike_ids)
        self._global_history.action(self.clustering)

    def move(self, clusters, group):
        self.cluster_meta.set('group', clusters, group)
        self._global_history.action(self.cluster_meta)

    def undo(self):
        self._global_history.undo()

    def redo(self):
        self._global_history.redo()
