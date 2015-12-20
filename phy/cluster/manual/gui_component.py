# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import logging

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from phy.gui.actions import Actions
from phy.gui.widgets import Table
from phy.io.array import Selector
from phy.utils import IPlugin

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
# Clustering GUI component
# -----------------------------------------------------------------------------

class ClusterView(Table):
    def __init__(self):
        super(ClusterView, self).__init__()
        self.add_styles('''
                        table tr[data-good='true'] {
                            color: #86D16D;
                        }
                        ''')


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
    shortcuts : dict

    GUI events
    ----------

    When this component is attached to a GUI, the GUI emits the following
    events:

    select(cluster_ids, selector)
        when clusters are selected
    cluster(up)
        when a merge or split happens
    request_save(spike_clusters, cluster_groups)
        when a save is requested by the user

    """

    default_shortcuts = {
        # Clustering.
        'merge': 'g',
        'split': 'k',

        # Move.
        'move_best_to_noise': 'alt+n',
        'move_best_to_mua': 'alt+m',
        'move_best_to_good': 'alt+g',

        'move_similar_to_noise': 'ctrl+n',
        'move_similar_to_mua': 'ctrl+m',
        'move_similar_to_good': 'ctrl+g',

        'move_all_to_noise': 'ctrl+alt+n',
        'move_all_to_mua': 'ctrl+alt+m',
        'move_all_to_good': 'ctrl+alt+g',

        # Wizard.
        'reset': 'ctrl+alt+space',
        'next': 'space',
        'previous': 'shift+space',
        'next_best': 'down',
        'previous_best': 'up',

        # Misc.
        'save': 'Save',
        'show_shortcuts': 'Save',
        'undo': 'Undo',
        'redo': ('ctrl+shift+z', 'ctrl+y'),
    }

    def __init__(self,
                 spike_clusters,
                 cluster_groups=None,
                 shortcuts=None,
                 ):

        self.gui = None

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Create Clustering and ClusterMeta.
        self.clustering = Clustering(spike_clusters)
        self.cluster_meta = create_cluster_meta(cluster_groups)
        self._global_history = GlobalHistory(process_ups=_process_ups)
        self._register_logging()

        # Create the spike selector.
        sc = self.clustering.spike_clusters
        spc = self.clustering.spikes_per_cluster
        self.selector = Selector(spike_clusters=sc,
                                 spikes_per_cluster=spc,
                                 )

        # Create the cluster views.
        self._create_cluster_views()
        self._add_default_columns()

        self.similarity_func = None

    # Internal methods
    # -------------------------------------------------------------------------

    def _register_logging(self):
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

    def _add_default_columns(self):
        # Default columns.
        @self.add_column(name='n_spikes')
        def n_spikes(cluster_id):
            return self.clustering.spike_counts[cluster_id]

        def skip(cluster_id):
            """Whether to skip that cluster."""
            return (self.cluster_meta.get('group', cluster_id)
                    in ('noise', 'mua'))
        self.add_column(skip, show=False)

        def good(cluster_id):
            """Good column for color."""
            return self.cluster_meta.get('group', cluster_id) == 'good'
        self.add_column(good, show=False)

        self._best = None

        def similarity(cluster_id):
            return self.similarity_func(cluster_id, self._best)
        self.similarity_view.add_column(similarity)

    def _create_actions(self, gui):
        self.actions = Actions(gui,
                               name='Clustering',
                               menu='&Clustering',
                               default_shortcuts=self.shortcuts)

        # Selection.
        self.actions.add(self.select, alias='c')
        self.actions.separator()

        # Clustering.
        self.actions.add(self.merge, alias='g')
        self.actions.add(self.split, alias='k')
        self.actions.separator()

        # Move.
        self.actions.add(self.move)

        for group in ('noise', 'mua', 'good'):
            self.actions.add(partial(self.move_best, group),
                             name='move_best_to_' + group,
                             docstring='Move the best clusters to %s.' % group)
            self.actions.add(partial(self.move_similar, group),
                             name='move_similar_to_' + group,
                             docstring='Move the similar clusters to %s.' %
                             group)
            self.actions.add(partial(self.move_all, group),
                             name='move_all_to_' + group,
                             docstring='Move all selected clusters to %s.' %
                             group)
        self.actions.separator()

        # Others.
        self.actions.add(self.undo)
        self.actions.add(self.redo)
        self.actions.add(self.save)

        # Wizard.
        self.actions.add(self.reset, menu='&Wizard')
        self.actions.add(self.next, menu='&Wizard')
        self.actions.add(self.previous, menu='&Wizard')
        self.actions.add(self.next_best, menu='&Wizard')
        self.actions.add(self.previous_best, menu='&Wizard')
        self.actions.separator()

    def _create_cluster_views(self):
        # Create the cluster view.
        self.cluster_view = ClusterView()
        self.cluster_view.build()

        # Create the similarity view.
        self.similarity_view = ClusterView()
        self.similarity_view.build()

        # Selection in the cluster view.
        @self.cluster_view.connect_
        def on_select(cluster_ids):
            # Emit GUI.select when the selection changes in the cluster view.
            self._emit_select(cluster_ids)
            # Pin the clusters and update the similarity view.
            self._update_similarity_view()

        # Selection in the similarity view.
        @self.similarity_view.connect_  # noqa
        def on_select(cluster_ids):
            # Select the clusters from both views.
            cluster_ids = self.cluster_view.selected + cluster_ids
            self._emit_select(cluster_ids)

        # Save the current selection when an action occurs.
        def on_request_undo_state(up):
            return {'selection': (self.cluster_view.selected,
                                  self.similarity_view.selected)}

        self.clustering.connect(on_request_undo_state)
        self.cluster_meta.connect(on_request_undo_state)

        self._update_cluster_view()

    def _update_cluster_view(self):
        """Initialize the cluster view with cluster data."""
        self.cluster_view.set_rows(self.clustering.cluster_ids)

    def _update_similarity_view(self):
        """Update the similarity view with matches for the specified
        clusters."""
        if not self.similarity_func:
            return
        selection = self.cluster_view.selected
        if not len(selection):
            return
        cluster_id = selection[0]
        self._best = cluster_id
        self.similarity_view.set_rows([c for c in self.clustering.cluster_ids
                                       if c not in selection])
        self.similarity_view.sort_by('similarity', 'desc')

    def _emit_select(self, cluster_ids):
        """Choose spikes from the specified clusters and emit the
        `select` event on the GUI."""
        logger.debug("Select clusters: %s.", ', '.join(map(str, cluster_ids)))
        if self.gui:
            self.gui.emit('select',
                          cluster_ids=cluster_ids,
                          selector=self.selector,
                          )

    # Public methods
    # -------------------------------------------------------------------------

    def add_column(self, func=None, name=None, show=True):
        if func is None:
            return lambda f: self.add_column(f, name=name, show=show)
        self.cluster_view.add_column(func, name=name, show=show)
        self.similarity_view.add_column(func, name=name, show=show)

    def set_default_sort(self, name, sort_dir='desc'):
        logger.debug("Set default sort `%s` %s.", name, sort_dir)
        # Set the default sort.
        self.cluster_view.set_default_sort(name, sort_dir)
        # Reset the cluster view.
        self._update_cluster_view()
        # Sort by the default sort.
        self.cluster_view.sort_by(name, sort_dir)

    def set_similarity_func(self, f):
        """Set the similarity function."""
        self.similarity_func = f

    def on_cluster(self, up):
        """Update the cluster views after clustering actions."""

        similar = self.similarity_view.selected

        # Reinitialize the cluster view if clusters have changed.
        if up.added:
            self._update_cluster_view()

        # Select all new clusters in view 1.
        if up.history == 'undo':
            # Select the clusters that were selected before the undone
            # action.
            clusters_0, clusters_1 = up.undo_state[0]['selection']
            self.cluster_view.select(clusters_0)
            self.similarity_view.select(clusters_1)
        elif up.added:
            self.select(up.added)
            if similar:
                self.similarity_view.next()
        elif up.metadata_changed:
            # Select next in similarity view if all moved are in that view.
            if set(up.metadata_changed) <= set(similar):

                # Update the cluster view, and select the clusters that
                # were selected before the action.
                selected = self.similarity_view.selected
                self._update_similarity_view()
                self.similarity_view.select(selected, do_emit=False)
                self.similarity_view.next()
            # Otherwise, select next in cluster view.
            else:
                # Update the cluster view, and select the clusters that
                # were selected before the action.
                selected = self.cluster_view.selected
                self._update_cluster_view()
                self.cluster_view.select(selected, do_emit=False)
                self.cluster_view.next()
                if similar:
                    self.similarity_view.next()

    def attach(self, gui):
        self.gui = gui
        gui.register(self, name='manual_clustering')

        # Create the actions.
        self._create_actions(gui)

        # Add the cluster views.
        gui.add_view(self.cluster_view, name='ClusterView')
        gui.add_view(self.similarity_view, name='SimilarityView')

        # Add the quality column in the cluster view.
        cs = gui.request('cluster_store')
        if cs:
            self.cluster_view.add_column(cs.max_waveform_amplitude,
                                         name='quality')
            self.set_default_sort('quality')
            self.set_similarity_func(cs.mean_masked_features_score)

        # Update the cluster views and selection when a cluster event occurs.
        self.gui.connect_(self.on_cluster)
        return self

    # Selection actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids):
        """Select a list of clusters."""
        # HACK: allow for `select(1, 2, 3)` in addition to `select([1, 2, 3])`
        # This makes it more convenient to select multiple clusters with
        # the snippet: `:c 1 2 3` instead of `:c 1,2,3`.
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
        # Update the cluster view selection.
        self.cluster_view.select(cluster_ids)

    @property
    def selected(self):
        return self.cluster_view.selected + self.similarity_view.selected

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None):
        """Merge the selected clusters."""
        if cluster_ids is None:
            cluster_ids = self.selected
        if len(cluster_ids or []) <= 1:
            return
        self.clustering.merge(cluster_ids)
        self._global_history.action(self.clustering)

    def split(self, spike_ids):
        """Split the selected spikes (NOT IMPLEMENTED YET)."""
        if len(spike_ids) == 0:
            return
        # TODO: connect to request_split emitted by view
        self.clustering.split(spike_ids)
        self._global_history.action(self.clustering)

    # Move actions
    # -------------------------------------------------------------------------

    def move(self, cluster_ids, group):
        """Move clusters to a group."""
        if len(cluster_ids) == 0:
            return
        self.cluster_meta.set('group', cluster_ids, group)
        self._global_history.action(self.cluster_meta)

    def move_best(self, group):
        """Move all selected best clusters to a group."""
        self.move(self.cluster_view.selected, group)

    def move_similar(self, group):
        """Move all selected similar clusters to a group."""
        self.move(self.similarity_view.selected, group)

    def move_all(self, group):
        """Move all selected clusters to a group."""
        self.move(self.selected, group)

    # Wizard actions
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset the wizard."""
        self._update_cluster_view()
        self.cluster_view.next()

    def next_best(self):
        """Select the next best cluster."""
        self.cluster_view.next()

    def previous_best(self):
        """Select the previous best cluster."""
        self.cluster_view.previous()

    def next(self):
        """Select the next cluster."""
        if not self.selected:
            self.cluster_view.next()
        else:
            self.similarity_view.next()

    def previous(self):
        """Select the previous cluster."""
        self.similarity_view.previous()

    # Other actions
    # -------------------------------------------------------------------------

    def undo(self):
        """Undo the last action."""
        self._global_history.undo()

    def redo(self):
        """Undo the last undone action."""
        self._global_history.redo()

    def save(self):
        """Save the manual clustering back to disk."""
        spike_clusters = self.clustering.spike_clusters
        groups = {c: self.cluster_meta.get('group', c) or 'unsorted'
                  for c in self.clustering.cluster_ids}
        self.gui.emit('request_save', spike_clusters, groups)


class ManualClusteringPlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):

        # Attach the manual clustering logic (wizard, merge, split,
        # undo stack) to the GUI.
        mc = ManualClustering(model.spike_clusters,
                              cluster_groups=model.cluster_groups,
                              )
        mc.attach(gui)
