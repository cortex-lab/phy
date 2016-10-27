# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial
import logging

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from phy.gui.qt import _show_box
from phy.gui.actions import Actions
from phy.gui.widgets import Table

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

    @property
    def state(self):
        return {'sort_by': self.current_sort}

    def set_state(self, state):
        sort_by, order = state.get('sort_by', (None, None))
        if sort_by:
            self.sort_by(sort_by, order)


class ManualClustering(object):
    """Component that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Parameters
    ----------

    spike_clusters : ndarray
    spikes_per_cluster : function `cluster_id -> spike_ids`
    cluster_groups : dictionary
    shortcuts : dict
    quality: func
    similarity: func

    GUI events
    ----------

    When this component is attached to a GUI, the GUI emits the following
    events:

    select(cluster_ids)
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

        'label': 'l',

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
                 spikes_per_cluster,
                 cluster_groups=None,
                 best_channel=None,
                 shortcuts=None,
                 quality=None,
                 similarity=None,
                 new_cluster_id=None,
                 ):

        self.gui = None
        self.quality = quality  # function cluster => quality
        self.similarity = similarity  # function cluster => [(cl, sim), ...]
        self.best_channel = best_channel  # function cluster_id => channel_id

        assert hasattr(spikes_per_cluster, '__call__')
        self.spikes_per_cluster = spikes_per_cluster

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Create Clustering and ClusterMeta.
        self.clustering = Clustering(spike_clusters,
                                     new_cluster_id=new_cluster_id)
        self.cluster_groups = cluster_groups or {}
        self.cluster_meta = create_cluster_meta(self.cluster_groups)
        self._global_history = GlobalHistory(process_ups=_process_ups)

        self.cluster_meta.add_field('next_cluster')

        @self.clustering.connect
        def on_cluster(up):
            """Register the next cluster in the list before the cluster
            view is updated."""
            if not up.added:
                return
            cluster = up.added[0]
            next_cluster = self.cluster_view.get_next_id()
            logger.debug("Register next_cluster to %d: %s",
                         cluster, next_cluster)
            self.cluster_meta.set('next_cluster', [cluster], next_cluster,
                                  add_to_stack=False)

        # NOTE: global on_cluster() occurs here.
        self._register_logging()

        # Create the cluster views.
        self._create_cluster_views()
        self._add_default_columns()

        self._best = None
        self._current_similarity_values = {}

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
            # Log changes.
            if up.history:
                logger.info(up.history.title() + " move.")
            else:
                logger.info("Change %s for clusters %s to %s.",
                            up.description,
                            ', '.join(map(str, up.metadata_changed)),
                            up.metadata_value)

            # Skip cluster metadata other than groups.
            if up.description != 'metadata_group':
                return

            # Update the original dictionary when groups change.
            for clu in up.metadata_changed:
                self.cluster_groups[clu] = up.metadata_value

            if self.gui:
                self.gui.emit('cluster', up)

    def _add_field_column(self, field):  # pragma: no cover
        """Add a column for a given label field."""
        @self.add_column(name=field)
        def get_my_label(cluster_id):
            return self.cluster_meta.get(field, cluster_id)

    def _add_default_columns(self):
        # Default columns.
        @self.add_column(name='n_spikes')
        def n_spikes(cluster_id):
            return len(self.spikes_per_cluster(cluster_id))

        self.add_column(self.best_channel, name='channel')

        @self.add_column(show=False)
        def skip(cluster_id):
            """Whether to skip that cluster."""
            return (self.cluster_meta.get('group', cluster_id)
                    in ('noise', 'mua'))

        @self.add_column(show=False)
        def good(cluster_id):
            """Good column for color."""
            return self.cluster_meta.get('group', cluster_id) == 'good'

        # Add columns for labels.
        for field in self.fields:  # pragma: no cover
            self._add_field_column(field)

        def similarity(cluster_id):
            # NOTE: there is a dictionary with the similarity to the current
            # best cluster. It is updated when the selection changes in the
            # cluster view. This is a bit of a hack: the HTML table expects
            # a function that returns a value for every row, but here we
            # cache all similarity view rows in self._current_similarity_values
            return self._current_similarity_values.get(cluster_id, 0)
        if self.similarity:
            self.similarity_view.add_column(similarity,
                                            name=self.similarity.__name__)

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

        # Label.
        self.actions.add(self.label, alias='l')
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
        logger.log(5, "Update the cluster view.")
        cluster_ids = [int(c) for c in self.clustering.cluster_ids]
        self.cluster_view.set_rows(cluster_ids)

    def _update_similarity_view(self):
        """Update the similarity view with matches for the specified
        clusters."""
        if not self.similarity:
            return
        selection = self.cluster_view.selected
        if not len(selection):
            return
        cluster_id = selection[0]
        cluster_ids = self.clustering.cluster_ids
        self._best = cluster_id
        logger.log(5, "Update the similarity view.")
        # This is a list of pairs (closest_cluster, similarity).
        similarities = self.similarity(cluster_id)
        # We save the similarity values wrt the currently-selected clusters.
        # Note that we keep the order of the output of the self.similary()
        # function.
        clusters_sim = OrderedDict([(int(cl), s) for (cl, s) in similarities])
        # List of similar clusters, remove non-existing ones.
        clusters = [c for c in clusters_sim.keys()
                    if c in cluster_ids]
        # The similarity view will use these values.
        self._current_similarity_values = clusters_sim
        # Set the rows of the similarity view.
        # TODO: instead of the self._current_similarity_values hack,
        # give the possibility to specify the values here (?).
        self.similarity_view.set_rows([c for c in clusters
                                       if c not in selection])

    def _emit_select(self, cluster_ids):
        """Choose spikes from the specified clusters and emit the
        `select` event on the GUI."""
        logger.debug("Select clusters: %s.", ', '.join(map(str, cluster_ids)))
        if self.gui:
            self.gui.emit('select', cluster_ids)

    # Public methods
    # -------------------------------------------------------------------------

    def add_column(self, func=None, name=None, show=True, default=False):
        if func is None:
            return lambda f: self.add_column(f, name=name, show=show,
                                             default=default)
        name = name or func.__name__
        assert name
        self.cluster_view.add_column(func, name=name, show=show)
        self.similarity_view.add_column(func, name=name, show=show)
        if default:
            self.set_default_sort(name)

    def set_default_sort(self, name, sort_dir='desc'):
        assert name
        logger.debug("Set default sort `%s` %s.", name, sort_dir)
        # Set the default sort.
        self.cluster_view.set_default_sort(name, sort_dir)
        # Reset the cluster view.
        self._update_cluster_view()
        # Sort by the default sort.
        self.cluster_view.sort_by(name, sort_dir)

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
            if up.description == 'assign':
                # NOTE: we reverse the order such that the last selected
                # cluster (with a new color) is the split cluster.
                added = up.added[::-1]
            else:
                added = up.added
            self.select(added)
            if similar:
                self.similarity_view.next()
        elif up.metadata_changed:
            # Select next in similarity view if all moved are in that view.
            if set(up.metadata_changed) <= set(similar):
                self._update_similarity_view()
                next_cluster = self.similarity_view.get_next_id()
                if next_cluster is not None:
                    self.similarity_view.select([next_cluster])
            # Otherwise, select next in cluster view.
            else:
                self._update_cluster_view()
                # Determine if there is a next cluster set from a
                # previous clustering action.
                cluster = up.metadata_changed[0]
                next_cluster = self.cluster_meta.get('next_cluster', cluster)
                logger.debug("Get next_cluster for %d: %s.",
                             cluster, next_cluster)
                # If there is not, fallback on the next cluster in the list.
                if next_cluster is None:
                    self.cluster_view.select([cluster], do_emit=False)
                    self.cluster_view.next()
                else:
                    self.cluster_view.select([next_cluster])

    def attach(self, gui):
        self.gui = gui

        # Create the actions.
        self._create_actions(gui)

        # Add the cluster views.
        gui.add_view(self.cluster_view, name='ClusterView')

        # Add the quality column in the cluster view.
        if self.quality:
            self.cluster_view.add_column(self.quality,
                                         name=self.quality.__name__,
                                         )

        # Update the cluster view and sort by n_spikes at the beginning.
        self._update_cluster_view()
        # if not self.quality:
        #     self.cluster_view.sort_by('n_spikes', 'desc')

        # Add the similarity view if there is a similarity function.
        if self.similarity:
            gui.add_view(self.similarity_view, name='SimilarityView')

        # Set the view state.
        cv = self.cluster_view
        cv.set_state(gui.state.get_view_state(cv))

        # Save the view state in the GUI state.
        @gui.connect_
        def on_close():
            gui.state.update_view_state(cv, cv.state)
            # NOTE: create_gui() already saves the state, but the event
            # is registered *before* we add all views.
            gui.state.save()

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

    def split(self, spike_ids=None, spike_clusters_rel=0):
        """Split the selected spikes."""
        if spike_ids is None:
            spike_ids = self.gui.emit('request_split')
            spike_ids = np.concatenate(spike_ids).astype(np.int64)
        if len(spike_ids) == 0:
            msg = ("You first need to select spikes in the feature "
                   "view with a few Ctrl+Click around the spikes "
                   "that you want to split.")
            _show_box(self.gui.dialog(msg))
            return
        self.clustering.split(spike_ids,
                              spike_clusters_rel=spike_clusters_rel)
        self._global_history.action(self.clustering)

    # Move actions
    # -------------------------------------------------------------------------

    @property
    def fields(self):
        """Tuple of label fields."""
        return tuple(f for f in self.cluster_meta.fields
                     if f not in ('group', 'next_cluster'))

    def get_labels(self, field):
        """Return the labels of all clusters, for a given field."""
        return {c: self.cluster_meta.get(field, c)
                for c in self.clustering.cluster_ids}

    def label(self, name, value, cluster_ids=None):
        """Assign a label to clusters."""
        if cluster_ids is None:
            cluster_ids = self.cluster_view.selected
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        if len(cluster_ids) == 0:
            return
        self.cluster_meta.set(name, cluster_ids, value)
        self._global_history.action(self.cluster_meta)

    def move(self, group, cluster_ids=None):
        """Move clusters to a group."""
        self.label('group', group, cluster_ids=cluster_ids)

    def move_best(self, group=None):
        """Move all selected best clusters to a group."""
        self.move(group, self.cluster_view.selected)

    def move_similar(self, group=None):
        """Move all selected similar clusters to a group."""
        self.move(group, self.similarity_view.selected)

    def move_all(self, group=None):
        """Move all selected clusters to a group."""
        self.move(group, self.selected)

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
        # List of tuples (field_name, dictionary).
        labels = [(field, self.get_labels(field)) for field in self.fields]
        # TODO: add option in add_field to declare a field unsavable.
        self.gui.emit('request_save', spike_clusters, groups, *labels)
