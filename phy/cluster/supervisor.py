# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import logging

import numpy as np
from six import string_types

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering
from phy.utils import EventEmitter, Bunch
from phy.gui.actions import Actions
from phy.gui.widgets import Table, HTMLWidget

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
# Action flow
# -----------------------------------------------------------------------------

class ActionFlow(EventEmitter):
    """Keep track of all actions and state changes, and defines how selections change
    after an action."""
    def __init__(self):
        super(ActionFlow, self).__init__()
        self._flow = []

    def add_state(
            self, cluster_ids=None, similar=None,
            next_cluster=None, next_similar=None):
        state = Bunch(
            type='state',
            cluster_ids=cluster_ids, similar=similar,
            next_cluster=next_cluster, next_similar=next_similar)
        self._flow.append(state)
        return state

    def update_current_state(
            self, cluster_ids=None, similar=None,
            next_cluster=None, next_similar=None):
        state = self.current()
        if not state or state.type != 'state':
            state = self.add_state()
        state.cluster_ids = state.cluster_ids if state.cluster_ids is not None else cluster_ids
        state.similar = state.similar if state.similar is not None else similar
        state.next_cluster = state.next_cluster if state.next_cluster is not None else next_cluster
        state.next_similar = state.next_similar if state.next_similar is not None else next_similar
        self._flow[-1] = state

    def _add_action(self, name, **kwargs):
        action = Bunch(type='action', name=name, **kwargs)
        self._flow.append(action)
        state = self.state_after(action)
        self._flow.append(state)
        return state

    def add_merge(self, cluster_ids=None, to=None):
        return self._add_action('merge', cluster_ids=cluster_ids, to=to)

    def add_split(self, old_cluster_ids=None, new_cluster_ids=None):
        return self._add_action(
            'split', old_cluster_ids=old_cluster_ids, new_cluster_ids=new_cluster_ids)

    def add_move(self, cluster_ids=None, group=None):
        return self._add_action(
            'move', cluster_ids=cluster_ids, group=group)

    def add_undo(self, up=None):
        return self._add_action('undo', up=up)

    def add_redo(self, up=None):
        return self._add_action('redo', up=up)

    def current(self):
        if self._flow:
            return self._flow[-1]

    def state_after(self, action):
        state = getattr(self, '_state_after_%s' % action.name)(action)
        state.cluster_ids = state.get('cluster_ids', [])
        state.similar = state.get('similar', [])
        state.next_cluster = state.get('next_cluster', None)
        state.next_similar = state.get('next_similar', None)
        return state

    def _previous_state(self, obj):
        try:
            i = self._flow.index(obj)
        except ValueError:
            return
        if i == 0:
            return
        for k in range(1, 10):
            previous = self._flow[i - k]
            if previous.type == 'state':
                return previous

    def _last_undo(self):
        for obj in self._flow[::-1]:
            if obj.type == 'action' and obj.name == 'undo':
                return obj

    def _state_after_merge(self, action):
        previous_state = self._previous_state(action)
        similar = previous_state.next_similar
        return Bunch(type='state', cluster_ids=[action.to], similar=[similar])

    def _state_after_split(self, action):
        return Bunch(type='state', cluster_ids=action.new_cluster_ids)

    def _state_after_move(self, action):
        state = self._previous_state(action)
        moved_clusters = set(action.cluster_ids)
        # If all moved clusters are in the cluster view, then move to the next
        # cluster in the cluster view.
        if moved_clusters <= set(state.cluster_ids):
            # Request the next similar cluster to the next best cluster.
            next_similar = self.emit('request_next_similar', cluster_id=state.next_cluster)
            if next_similar:
                return Bunch(type='state',
                             cluster_ids=[state.next_cluster],
                             similar=[next_similar[0]])
            else:
                return Bunch(type='state',
                             cluster_ids=[state.next_cluster])
        # Otherwise, select the next one in the similarity view.
        elif moved_clusters <= set(state.similar):
            return Bunch(type='state', cluster_ids=state.cluster_ids,
                         similar=[state.next_similar])

    def _state_after_undo(self, action):
        return self._previous_state(self._previous_state(action))

    def _state_after_redo(self, action):
        undo = self._last_undo()
        if undo:
            return self._previous_state(undo)

    def to_json(self):
        return


# -----------------------------------------------------------------------------
# Cluster view and similarity view
# -----------------------------------------------------------------------------

class ClusterView(Table):
    def __init__(self, data=None):
        HTMLWidget.__init__(self, title='ClusterView')
        self._set_styles()

        # TODO: custom columns
        columns = ['id', 'n_spikes']
        assert columns[0] == 'id'

        # Allow to have <tr data_group="good"> etc. which allows for CSS styling.
        value_names = columns + [{'data': ['group']}]
        self._init_table(columns=columns, value_names=value_names, data=data)

    def _set_styles(self):
        self.builder.add_style('''
            table tr[data-group='good'] {
                color: #86D16D;
            }

            table tr[data-group='mua'], table tr[data-group='noise'] {
                color: #888;
            }
            ''')

    def get_state(self, callback=None):
        self.get_current_sort(lambda sort: callback({'current_sort': tuple(sort)}))

    def set_state(self, state):
        sort_by, sort_dir = state.get('current_sort', (None, None))
        if sort_by:
            self.sort_by(sort_by, sort_dir)


class SimilarityView(ClusterView):
    """Must connect request_similar_clusters."""
    def __init__(self, data=None):
        HTMLWidget.__init__(self, title='SimilarityView')
        self._set_styles()
        columns = ['id', 'n_spikes', 'similarity']
        value_names = columns + [{'data': ['group']}]
        self._init_table(columns=columns, value_names=value_names, data=data)

    def reset(self, cluster_id):
        similar = self.emit('request_similar_clusters', cluster_id)
        # Clear the table.
        self.remove_all()
        if similar:
            self.add(similar[0])


# -----------------------------------------------------------------------------
# ActionCreator
# -----------------------------------------------------------------------------

class ActionCreator(EventEmitter):
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

    def add(self, name, **kwargs):
        self.actions.add(partial(self.emit, 'action', name), name=name, **kwargs)

    def separator(self, **kwargs):
        self.actions.separator(**kwargs)

    def attach(self, gui):
        self.actions = Actions(gui,
                               name='Clustering',
                               menu='&Clustering',
                               default_shortcuts=self.default_shortcuts)

        # Selection.
        self.add('select', alias='c', docstring='Select some clusters.')
        self.separator()

        self.add('undo', docstring='Undo the last action.')
        self.add('redo', docstring='Redo the last undone action.')
        self.separator()

        # Clustering.
        self.add('merge', alias='g', docstring='Merge the selected clusters.')
        self.add('split', alias='k', docstring='Create a new cluster out of the selected spikes')
        self.separator()

        # Move.
        self.add('move', docstring='Move some clusters to a group.')
        self.separator()

        for group in ('noise', 'mua', 'good'):
            self.add('move_best_to_' + group,
                     docstring='Move the best clusters to %s.' % group)
            self.add('move_similar_to_' + group,
                     docstring='Move the similar clusters to %s.' %
                     group)
            self.add('move_all_to_' + group,
                     docstring='Move all selected clusters to %s.' %
                     group)
            self.separator()

        # Label.
        self.add('label', alias='l', docstring='Label the selected clusters.')

        # Others.
        self.add('save', menu='&File', docstring='Save all pending changes.')

        # Wizard.
        self.add('reset', menu='&Wizard', docstring='Reset the wizard.')
        self.separator(menu='&Wizard')
        self.add('next', menu='&Wizard', docstring='Select the next similar cluster.')
        self.add('previous', menu='&Wizard', docstring='Select the previous similar cluster.')
        self.separator(menu='&Wizard')
        self.add('next_best', menu='&Wizard', docstring='Select the next best cluster.')
        self.add('previous_best', menu='&Wizard', docstring='Select the previous best cluster.')
        self.separator(menu='&Wizard')


# -----------------------------------------------------------------------------
# Clustering GUI component
# -----------------------------------------------------------------------------

class Supervisor(EventEmitter):
    """Component that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Parameters
    ----------

    spike_clusters : ndarray
    cluster_groups : dictionary
    quality: func
    similarity: func
    new_cluster_id: func
    context: Context instance

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

    def __init__(self,
                 spike_clusters=None,
                 cluster_groups=None,
                 quality=None,
                 similarity=None,
                 new_cluster_id=None,
                 context=None,
                 ):
        super(Supervisor, self).__init__()
        self.context = context
        self.quality = quality or self.n_spikes  # function cluster => quality
        self.similarity = similarity  # function cluster => [(cl, sim), ...]

        # Create Clustering and ClusterMeta.
        # Load the cached spikes_per_cluster array.
        spc = context.load('spikes_per_cluster') if context else None
        self.clustering = Clustering(spike_clusters,
                                     spikes_per_cluster=spc,
                                     new_cluster_id=new_cluster_id)
        # Cache the spikes_per_cluster array.
        self._save_spikes_per_cluster()

        # Create the ClusterMeta instance.
        self.cluster_meta = create_cluster_meta(cluster_groups or {})

        # Create the GlobalHistory instance.
        self._global_history = GlobalHistory(process_ups=_process_ups)

        # Create the Action Flow instance.
        self.action_flow = ActionFlow()

        # Create The Action Creator instance.
        self.action_creator = ActionCreator()
        self.action_creator.connect(self._on_action, event='action')

        self._create_views()
        # Save the next cluster in ClusterMeta.
        # self.cluster_meta.add_field('next_cluster')
        # self.clustering.connect(self._register_next_cluster, event='on_cluster')

        # Log the actions.
        self.clustering.connect(self._log_action, event='cluster')
        self.cluster_meta.connect(self._log_action_meta, event='cluster')

        # Raise the global cluster event.
        self.clustering.connect(partial(self.emit, 'cluster'), event='cluster')
        self.cluster_meta.connect(partial(self.emit, 'cluster'), event='cluster')

    # Internal methods
    # -------------------------------------------------------------------------

    def _save_spikes_per_cluster(self):
        if not self.context:
            return
        self.context.save('spikes_per_cluster',
                          self.clustering.spikes_per_cluster,
                          kind='pickle',
                          )

    def _log_action(self, up):
        if up.history:
            logger.info(up.history.title() + " cluster assign.")
        elif up.description == 'merge':
            logger.info("Merge clusters %s to %s.",
                        ', '.join(map(str, up.deleted)),
                        up.added[0])
        else:
            logger.info("Assigned %s spikes.", len(up.spike_ids))
        #self.emit('cluster', up)

    def _log_action_meta(self, up):
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

    def _save_new_cluster_id(self, up):
        # Save the new cluster id on disk.
        new_cluster_id = self.clustering.new_cluster_id()
        if self.context:
            logger.debug("Save the new cluster id: %d.", new_cluster_id)
            self.context.save('new_cluster_id',
                              dict(new_cluster_id=new_cluster_id))

    def _save_gui_state(self, gui):
        gui.state.update_view_state(self.cluster_view, self.cluster_view.state)
        # NOTE: create_gui() already saves the state, but the event
        # is registered *before* we add all views.
        gui.state.save()

    def n_spikes(self, cluster_id):
        return len(self.clustering.spikes_per_cluster[cluster_id])

    def _get_similar_clusters(self, cluster_id):
        sim = self.similarity(cluster_id)
        data = [dict(similarity=s, **self._get_cluster_info(c))
                for c, s in sim]
        return data

    def _get_cluster_info(self, cluster_id):
        return {'id': cluster_id,
                'n_spikes': self.n_spikes(cluster_id),
                'quality': self.quality(cluster_id),
                }

    def _create_views(self):
        data = [self._get_cluster_info(cluster_id) for cluster_id in self.clustering.cluster_ids]
        self.cluster_view = ClusterView(data)
        self.cluster_view.connect_(self._clusters_selected, event='select')

        self.similarity_view = SimilarityView()
        self.similarity_view.connect_(self._get_similar_clusters, event='request_similar_clusters')
        self.similarity_view.connect_(self._similar_selected, event='select')

    def _clusters_added(self, cluster_ids):
        data = [self._get_cluster_info(cluster_id) for cluster_id in cluster_ids]
        self.cluster_view.add(data)
        self.similarity_view.add(data)

    def _clusters_removed(self, cluster_ids):
        self.cluster_view.remove(cluster_ids)
        self.similarity_view.remove(cluster_ids)

    def _cluster_groups_changed(self, cluster_ids):
        data = [{'id': cluster_id, 'group': self.cluster_meta.get('group', cluster_id)}
                for cluster_id in cluster_ids]
        self.cluster_view.change(data)
        self.similarity_view.change(data)

    def _clusters_selected(self, cluster_ids):
        self.action_flow.update_current_state(cluster_ids=cluster_ids)
        self.cluster_view.get_next_id(
            lambda next_cluster: self.action_flow.update_current_state(next_cluster=next_cluster))
        self.similarity_view.get_selected(
            lambda similar: self.emit('select', cluster_ids + similar))

    def _similar_selected(self, similar):
        self.action_flow.update_current_state(similar=similar)
        self.similarity_view.get_next_id(
            lambda next_similar: self.action_flow.update_current_state(next_similar=next_similar))
        self.cluster_view.get_selected(
            lambda cluster_ids: self.emit('select', cluster_ids + similar))

    def _on_action(self, name):
        return getattr(self, name)()

    def _select_after_action(self):
        state = self.action_flow.current()
        if not state.type == 'state':
            return
        if state.cluster_ids:
            self.cluster_view.select(state.cluster_ids)
        if state.similar:
            self.similarity_view.select(state.similar)

    def _after_action(self, up):
        # Update the views with the old and new clusters.
        self._clusters_added(up.added)
        self._clusters_removed(up.removed)

        # Prepare the next selection after the action.
        if up.description == 'merge':
            self.action_flow.add_merge(up.removed, up.added[0])
        elif up.description == 'assign':
            self.action_flow.add_split(old_cluster_ids=up.removed,
                                       new_cluster_ids=up.added)
        elif up.description == 'metadata_changed':
            self._cluster_groups_changed(up.metadata_changed)
            self.action_flow.add_move(up.metadata_changed, up.metadata_value)
        elif up.description == 'undo':
            self.action_flow.add_undo(up)
        elif up.description == 'redo':
            self.action_flow.add_redo(up)

        # Raise
        self.emit('cluster', up)

        # Make the new selection.
        self._select_after_action()

    def attach(self, gui):
        self.cluster_view.set_state(gui.state.get_view_state(self.cluster_view))
        gui.add_view(self.cluster_view)

        gui.add_view(self.similarity_view)

        self.action_creator.attach(gui)
        # TODO: gui should raise events too

    # Selection actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids):
        """Select a list of clusters."""
        # HACK: allow for `select(1, 2, 3)` in addition to `select([1, 2, 3])`
        # This makes it more convenient to select multiple clusters with
        # the snippet: `:c 1 2 3` instead of `:c 1,2,3`.
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
        # Remove non-existing clusters from the selection.
        #cluster_ids = self._keep_existing_clusters(cluster_ids)
        # Update the cluster view selection.
        self.cluster_view.select(cluster_ids)

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None, to=None):
        """Merge the selected clusters."""
        if cluster_ids is None:
            cluster_ids = self.selected
        if len(cluster_ids or []) <= 1:
            return
        self.clustering.merge(cluster_ids, to=to)
        self._global_history.action(self.clustering)

    def split(self, spike_ids=None, spike_clusters_rel=0):
        """Split the selected spikes."""
        if spike_ids is None:
            spike_ids = self.emit('request_split', single=True)
            spike_ids = np.asarray(spike_ids, dtype=np.int64)
            assert spike_ids.dtype == np.int64
            assert spike_ids.ndim == 1
        if len(spike_ids) == 0:
            msg = ("You first need to select spikes in the feature "
                   "view with a few Ctrl+Click around the spikes "
                   "that you want to split.")
            self.emit('error', msg)
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
        """Assign a label to clusters.

        Example: `quality 3`

        """
        if cluster_ids is None:
            cluster_ids = self.cluster_view.selected
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        if len(cluster_ids) == 0:
            return
        self.cluster_meta.set(name, cluster_ids, value)
        self._global_history.action(self.cluster_meta)

    def move(self, group, cluster_ids=None):
        """Assign a group to some clusters.

        Example: `good`

        """
        if isinstance(cluster_ids, string_types):
            logger.warn("The list of clusters should be a list of integers, "
                        "not a string.")
            return
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
        labels = [(field, self.get_labels(field))
                  for field in self.cluster_meta.fields
                  if field not in ('next_cluster')]
        # TODO: add option in add_field to declare a field unsavable.
        self.emit('request_save', spike_clusters, groups, *labels)
        # Cache the spikes_per_cluster array.
        self._save_spikes_per_cluster()
