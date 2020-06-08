# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import inspect
import logging

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering

from phylib.utils import Bunch, emit, connect, unconnect
from phy.gui.actions import Actions
from phy.gui.qt import _block, set_busy, _wait
from phy.gui.widgets import Table, HTMLWidget, _uniq, Barrier

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
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


def _ensure_all_ints(l):
    if (l is None or l == []):
        return
    for i in range(len(l)):
        l[i] = int(l[i])


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------

class TaskLogger(object):
    """Internal object that gandles all clustering actions and the automatic actions that
    should follow as part of the "wizard"."""

    # Whether to auto select next clusters after a merge.
    auto_select_after_action = False

    def __init__(self, cluster_view=None, similarity_view=None, supervisor=None):
        self.cluster_view = cluster_view
        self.similarity_view = similarity_view
        self.supervisor = supervisor
        self._processing = False
        # List of tasks that have completed.
        self._history = []
        # Tasks that have yet to be performed.
        self._queue = []

    def enqueue(self, sender, name, *args, output=None, **kwargs):
        """Enqueue an action, which has a sender, a function name, a list of arguments,
        and an optional output."""
        logger.log(
            5, "Enqueue %s %s %s %s (%s)", sender.__class__.__name__, name, args, kwargs, output)
        self._queue.append((sender, name, args, kwargs))

    def dequeue(self):
        """Dequeue the oldest item in the queue."""
        return self._queue.pop(0) if self._queue else None

    def _callback(self, task, output):
        """Called after the execution of an action in the queue.

        Will add the action to the history, with its input, enqueue subsequent actions, and
        ensure these actions are immediately executed.

        """
        # Log the task and its output.
        self._log(task, output)
        # Find the post tasks after that task has completed, and enqueue them.
        self.enqueue_after(task, output)
        # Loop.
        self.process()

    def _eval(self, task):
        """Evaluate a task and call a callback function."""
        sender, name, args, kwargs = task
        logger.log(5, "Calling %s.%s(%s)", sender.__class__.__name__, name, args, kwargs)
        f = getattr(sender, name)
        callback = partial(self._callback, task)
        argspec = inspect.getfullargspec(f)
        argspec = argspec.args + argspec.kwonlyargs
        if 'callback' in argspec:
            f(*args, **kwargs, callback=callback)
        else:
            # HACK: use on_cluster event instead of callback.
            def _cluster_callback(tsender, up):
                self._callback(task, up)
            connect(_cluster_callback, event='cluster', sender=self.supervisor)
            f(*args, **kwargs)
            unconnect(_cluster_callback)

    def process(self):
        """Process all tasks in queue."""
        self._processing = True
        task = self.dequeue()
        if not task:
            self._processing = False
            return
        # Process the first task in queue, or stop if the queue is empty.
        self._eval(task)

    def enqueue_after(self, task, output):
        """Enqueue tasks after a given action."""
        sender, name, args, kwargs = task
        f = lambda *args, **kwargs: logger.log(5, "No method _after_%s", name)
        getattr(self, '_after_%s' % name, f)(task, output)

    def _after_merge(self, task, output):
        """Tasks that should follow a merge."""
        merged, to = output.deleted, output.added[0]
        cluster_ids, next_cluster, similar, next_similar = self.last_state()
        # Update views after cluster_view.select event only if there is no similar clusters.
        # Otherwise, this is only the similarity_view that will raise the select event leading
        # to view updates.
        do_select_new = self.auto_select_after_action and similar is not None
        self.enqueue(self.cluster_view, 'select', [to], update_views=not do_select_new)
        if do_select_new:  # pragma: no cover
            if set(merged).intersection(similar) and next_similar is not None:
                similar = [next_similar]
            self.enqueue(self.similarity_view, 'select', similar)

    def _after_split(self, task, output):
        """Tasks that should follow a split."""
        self.enqueue(self.cluster_view, 'select', output.added)

    def _get_clusters(self, which):
        cluster_ids, next_cluster, similar, next_similar = self.last_state()
        if which == 'all':
            return _uniq(cluster_ids + similar)
        elif which == 'best':
            return cluster_ids
        elif which == 'similar':
            return similar
        return which

    def _after_move(self, task, output):
        """Tasks that should follow a move."""
        which = output.metadata_changed
        moved = set(self._get_clusters(which))
        cluster_ids, next_cluster, similar, next_similar = self.last_state()
        cluster_ids = set(cluster_ids or ())
        similar = set(similar or ())
        # Move best.
        if moved <= cluster_ids:
            self.enqueue(self.cluster_view, 'next')
        # Move similar.
        elif moved <= similar:
            self.enqueue(self.similarity_view, 'next')
        # Move all.
        else:
            self.enqueue(self.cluster_view, 'next')
            self.enqueue(self.similarity_view, 'next')

    def _after_undo(self, task, output):
        """Task that should follow an undo."""
        last_action = self.last_task(name_not_in=('select', 'next', 'previous', 'undo', 'redo'))
        self._select_state(self.last_state(last_action))

    def _after_redo(self, task, output):
        """Task that should follow an redo."""
        last_undo = self.last_task('undo')
        # Select the last state before the last undo.
        self._select_state(self.last_state(last_undo))

    def _select_state(self, state):
        """Enqueue select actions when a state (selected clusters and similar clusters) is set."""
        cluster_ids, next_cluster, similar, next_similar = state
        self.enqueue(
            self.cluster_view, 'select', cluster_ids, update_views=False if similar else True)
        if similar:
            self.enqueue(self.similarity_view, 'select', similar)

    def _log(self, task, output):
        """Add a completed task to the history stack."""
        sender, name, args, kwargs = task
        assert sender
        assert name
        logger.log(
            5, "Log %s %s %s %s (%s)", sender.__class__.__name__, name, args, kwargs, output)
        args = [a.tolist() if isinstance(a, np.ndarray) else a for a in args]
        task = (sender, name, args, kwargs, output)
        # Avoid successive duplicates (even if sender is different).
        if not self._history or self._history[-1][1:] != task[1:]:
            self._history.append(task)

    def log(self, sender, name, *args, output=None, **kwargs):
        """Add a completed task to the history stack."""
        self._log((sender, name, args, kwargs), output)

    def last_task(self, name=None, name_not_in=()):
        """Return the last executed task."""
        for (sender, name_, args, kwargs, output) in reversed(self._history):
            if (name and name_ == name) or (name_not_in and name_ and name_ not in name_not_in):
                assert name_
                return (sender, name_, args, kwargs, output)

    def last_state(self, task=None):
        """Return (cluster_ids, next_cluster, similar, next_similar)."""
        cluster_state = (None, None)
        similarity_state = (None, None)
        h = self._history
        # Last state until the passed task, if applicable.
        if task:
            i = self._history.index(task)
            h = self._history[:i]
        for (sender, name, args, kwargs, output) in reversed(h):
            # Last selection is cluster view selection: return the state.
            if (sender == self.similarity_view and similarity_state == (None, None) and
                    name in ('select', 'next', 'previous')):
                similarity_state = (output['selected'], output['next']) if output else (None, None)
            if (sender == self.cluster_view and
                    cluster_state == (None, None) and
                    name in ('select', 'next', 'previous')):
                cluster_state = (output['selected'], output['next']) if output else (None, None)
                return (*cluster_state, *similarity_state)

    def show_history(self):
        """Show the history stack."""
        print("=== History ===")
        for sender, name, args, kwargs, output in self._history:
            print(
                '{: <24} {: <8}'.format(sender.__class__.__name__, name), *args, output, kwargs)

    def has_finished(self):
        """Return whether the queue has finished being processed."""
        return len(self._queue) == 0 and not self._processing


# -----------------------------------------------------------------------------
# Cluster view and similarity view
# -----------------------------------------------------------------------------

_CLUSTER_VIEW_STYLES = '''
table tr[data-group='good'] {
    color: #86D16D;
}

table tr[data-group='mua'] {
    color: #afafaf;
}

table tr[data-group='noise'] {
    color: #777;
}
'''


class ClusterView(Table):
    """Display a table of all clusters with metrics and labels as columns. Derive from Table.

    Constructor
    -----------

    parent : Qt widget
    data : list
        List of dictionaries mapping fields to values.
    columns : list
        List of columns in the table.
    sort : 2-tuple
        Initial sort of the table as a pair (column_name, order), where order is
        either `asc` or `desc`.

    """

    _required_columns = ('n_spikes',)
    _view_name = 'cluster_view'
    _styles = _CLUSTER_VIEW_STYLES

    def __init__(self, *args, data=None, columns=(), sort=None):
        # NOTE: debounce select events.
        HTMLWidget.__init__(
            self, *args, title=self.__class__.__name__, debounce_events=('select',))
        self._set_styles()
        self._reset_table(data=data, columns=columns, sort=sort)

    def _reset_table(self, data=None, columns=(), sort=None):
        """Recreate the table with specified columns, data, and sort."""
        emit(self._view_name + '_init', self)
        # Ensure 'id' is the first column.
        if 'id' in columns:
            columns.remove('id')
        columns = ['id'] + list(columns)
        # Add required columns if needed.
        for col in self._required_columns:
            if col not in columns:
                columns += [col]
            assert col in columns
        assert columns[0] == 'id'

        # Allow to have <tr data_group="good"> etc. which allows for CSS styling.
        value_names = columns + [{'data': ['group']}]
        # Default sort.
        sort = sort or ('n_spikes', 'desc')
        self._init_table(columns=columns, value_names=value_names, data=data, sort=sort)

    def _set_styles(self):
        self.builder.add_style(self._styles)

    @property
    def state(self):
        """Return the cluster view state, with the current sort and selection."""

        b = Barrier()
        self.get_current_sort(b('current_sort'))
        self.get_selected(b('selected'))
        b.wait()

        current_sort = tuple(b.result('current_sort')[0][0] or (None, None))
        selected = b.result('selected')[0][0]

        return {
            'current_sort': current_sort,
            'selected': selected,
        }

    def set_state(self, state):
        """Set the cluster view state, with a specified sort."""
        sort_by, sort_dir = state.get('current_sort', (None, None))
        if sort_by:
            self.sort_by(sort_by, sort_dir)
        selected = state.get('selected', [])
        if selected:
            self.select(selected)


class SimilarityView(ClusterView):
    """Display a table of clusters with metrics and labels as columns, and an additional
    similarity column.

    This view displays clusters similar to the clusters currently selected
    in the cluster view.

    Events
    ------

    * request_similar_clusters(cluster_id)

    """

    _required_columns = ('n_spikes', 'similarity')
    _view_name = 'similarity_view'

    def set_selected_index_offset(self, n):
        """Set the index of the selected cluster, used for correct coloring in the similarity
        view."""
        self.eval_js('table._setSelectedIndexOffset(%d);' % n)

    def reset(self, cluster_ids):
        """Recreate the similarity view, given the selected clusters in the cluster view."""
        if not len(cluster_ids):
            return
        similar = emit('request_similar_clusters', self, cluster_ids[-1])
        # Clear the table.
        if similar:
            self.remove_all_and_add(
                [cl for cl in similar[0] if cl['id'] not in cluster_ids])
        else:  # pragma: no cover
            self.remove_all()
        return similar


# -----------------------------------------------------------------------------
# ActionCreator
# -----------------------------------------------------------------------------

class ActionCreator(object):
    """Companion class to the Supervisor that manages the related GUI actions."""

    default_shortcuts = {
        # Clustering.
        'merge': 'g',
        'split': 'k',

        'label': 'l',

        # Move.
        'move_best_to_noise': 'alt+n',
        'move_best_to_mua': 'alt+m',
        'move_best_to_good': 'alt+g',
        'move_best_to_unsorted': 'alt+u',

        'move_similar_to_noise': 'ctrl+n',
        'move_similar_to_mua': 'ctrl+m',
        'move_similar_to_good': 'ctrl+g',
        'move_similar_to_unsorted': 'ctrl+u',

        'move_all_to_noise': 'ctrl+alt+n',
        'move_all_to_mua': 'ctrl+alt+m',
        'move_all_to_good': 'ctrl+alt+g',
        'move_all_to_unsorted': 'ctrl+alt+u',

        # Wizard.
        'first': 'home',
        'last': 'end',
        'reset': 'ctrl+alt+space',
        'next': 'space',
        'previous': 'shift+space',
        'unselect_similar': 'backspace',
        'next_best': 'down',
        'previous_best': 'up',

        # Misc.
        'undo': 'ctrl+z',
        'redo': ('ctrl+shift+z', 'ctrl+y'),

        'clear_filter': 'esc',
    }

    default_snippets = {
        'merge': 'g',
        'split': 'k',
        'label': 'l',
        'select': 'c',
        'filter': 'f',
        'sort': 's',
    }

    def __init__(self, supervisor=None):
        self.supervisor = supervisor

    def add(self, which, name, **kwargs):
        """Add an action to a given menu."""
        # This special keyword argument lets us use a different name for the
        # action and the event name/method (used for different move flavors).
        method_name = kwargs.pop('method_name', name)
        method_args = kwargs.pop('method_args', ())
        emit_fun = partial(emit, 'action', self, method_name, *method_args)
        f = getattr(self.supervisor, method_name, None)
        docstring = inspect.getdoc(f) if f else name
        if not kwargs.get('docstring', None):
            kwargs['docstring'] = docstring
        getattr(self, '%s_actions' % which).add(emit_fun, name=name, **kwargs)

    def attach(self, gui):
        """Attach the GUI and create the menus."""
        # Create the menus.
        ds = self.default_shortcuts
        dsp = self.default_snippets
        self.edit_actions = Actions(
            gui, name='Edit', menu='&Edit', insert_menu_before='&View',
            default_shortcuts=ds, default_snippets=dsp)
        self.select_actions = Actions(
            gui, name='Select', menu='Sele&ct', insert_menu_before='&View',
            default_shortcuts=ds, default_snippets=dsp)

        # Create the actions.
        self._create_edit_actions()
        self._create_select_actions()
        self._create_toolbar(gui)

    def _create_edit_actions(self):
        w = 'edit'
        self.add(w, 'undo', set_busy=True, icon='f0e2')
        self.add(w, 'redo', set_busy=True, icon='f01e')
        self.edit_actions.separator()

        # Clustering.
        self.add(w, 'merge', set_busy=True, icon='f247')
        self.add(w, 'split', set_busy=True)
        self.edit_actions.separator()

        # Move.
        self.add(w, 'move', prompt=True, n_args=2)
        for which in ('best', 'similar', 'all'):
            for group in ('noise', 'mua', 'good', 'unsorted'):
                self.add(
                    w, 'move_%s_to_%s' % (which, group),
                    method_name='move',
                    method_args=(group, which),
                    submenu='Move %s to' % which,
                    docstring='Move %s to %s.' % (which, group))
        self.edit_actions.separator()

        # Label.
        self.add(w, 'label', prompt=True, n_args=2)
        self.edit_actions.separator()

    def _create_select_actions(self):
        w = 'select'

        # Selection.
        self.add(w, 'select', prompt=True, n_args=1)
        self.add(w, 'unselect_similar')
        self.select_actions.separator()

        # Sort and filter
        self.add(w, 'filter', prompt=True, n_args=1)
        self.add(w, 'sort', prompt=True, n_args=1)
        self.add(w, 'clear_filter')

        # Sort by:
        for column in getattr(self.supervisor, 'columns', ()):
            self.add(
                w, 'sort_by_%s' % column.lower(), method_name='sort', method_args=(column,),
                docstring='Sort by %s' % column,
                submenu='Sort by', alias='s%s' % column.replace('_', '')[:2])

        self.select_actions.separator()
        self.add(w, 'first')
        self.add(w, 'last')

        self.select_actions.separator()

        self.add(w, 'reset_wizard', icon='f015')
        self.select_actions.separator()

        self.add(w, 'next', icon='f061')
        self.add(w, 'previous', icon='f060')
        self.select_actions.separator()

        self.add(w, 'next_best', icon='f0a9')
        self.add(w, 'previous_best', icon='f0a8')
        self.select_actions.separator()

    def _create_toolbar(self, gui):
        gui._toolbar.addAction(self.edit_actions.get('undo'))
        gui._toolbar.addAction(self.edit_actions.get('redo'))
        gui._toolbar.addSeparator()
        gui._toolbar.addAction(self.select_actions.get('reset_wizard'))
        gui._toolbar.addAction(self.select_actions.get('previous_best'))
        gui._toolbar.addAction(self.select_actions.get('next_best'))
        gui._toolbar.addAction(self.select_actions.get('previous'))
        gui._toolbar.addAction(self.select_actions.get('next'))
        gui._toolbar.addSeparator()
        gui._toolbar.show()


# -----------------------------------------------------------------------------
# Clustering GUI component
# -----------------------------------------------------------------------------

def _is_group_masked(group):
    return group in ('noise', 'mua')


class Supervisor(object):
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
            self, spike_clusters=None, cluster_groups=None, cluster_metrics=None,
            cluster_labels=None, similarity=None, new_cluster_id=None, sort=None, context=None):
        super(Supervisor, self).__init__()
        self.context = context
        self.similarity = similarity  # function cluster => [(cl, sim), ...]
        self.actions = None  # will be set when attaching the GUI
        self._is_dirty = None
        self._sort = sort  # Initial sort requested in the constructor

        # Cluster metrics.
        # This is a dict {name: func cluster_id => value}.
        self.cluster_metrics = cluster_metrics or {}
        self.cluster_metrics['n_spikes'] = self.n_spikes

        # Cluster labels.
        # This is a dict {name: {cl: value}}
        self.cluster_labels = cluster_labels or {}

        self.columns = ['id']  # n_spikes comes from cluster_metrics
        self.columns += list(self.cluster_metrics.keys())
        self.columns += [
            label for label in self.cluster_labels.keys()
            if label not in self.columns + ['group']]

        # Create Clustering and ClusterMeta.
        # Load the cached spikes_per_cluster array.
        spc = context.load('spikes_per_cluster') if context else None
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

        # Create The Action Creator instance.
        self.action_creator = ActionCreator(self)
        connect(self._on_action, event='action', sender=self.action_creator)

        # Log the actions.
        connect(self._log_action, event='cluster', sender=self.clustering)
        connect(self._log_action_meta, event='cluster', sender=self.cluster_meta)

        # Raise supervisor.cluster
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

        self._is_busy = False

    # Internal methods
    # -------------------------------------------------------------------------

    def _save_spikes_per_cluster(self):
        """Cache on the disk the dictionary with the spikes belonging to each cluster."""
        if not self.context:
            return
        self.context.save('spikes_per_cluster', self.clustering.spikes_per_cluster, kind='pickle')

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
        new_cluster_id = self.clustering.new_cluster_id()
        if self.context:
            logger.log(5, "Save the new cluster id: %d.", new_cluster_id)
            self.context.save('new_cluster_id', dict(new_cluster_id=new_cluster_id))

    def _save_gui_state(self, gui):
        """Save the GUI state with the cluster view and similarity view."""
        gui.state.update_view_state(self.cluster_view, self.cluster_view.state)

    def _get_similar_clusters(self, sender, cluster_id):
        """Return the clusters similar to a given cluster."""
        sim = self.similarity(cluster_id) or []
        # Only keep existing clusters.
        clusters_set = set(self.clustering.cluster_ids)
        data = [
            dict(similarity='%.3f' % s, **self.get_cluster_info(c))
            for c, s in sim if c in clusters_set]
        return data

    def get_cluster_info(self, cluster_id, exclude=()):
        """Return the data associated to a given cluster."""
        out = {'id': cluster_id}
        # Cluster metrics.
        for key, func in self.cluster_metrics.items():
            out[key] = func(cluster_id)
        # Cluster meta.
        for key in self.cluster_meta.fields:
            # includes group
            out[key] = self.cluster_meta.get(key, cluster_id)
        out['is_masked'] = _is_group_masked(out.get('group', None))
        return {k: v for k, v in out.items() if k not in exclude}

    def _create_views(self, gui=None, sort=None):
        """Create the cluster view and similarity view."""

        sort = sort or self._sort  # comes from either the GUI state or constructor

        # Create the cluster view.
        self.cluster_view = ClusterView(
            gui, data=self.cluster_info, columns=self.columns, sort=sort)
        # Update the action flow and similarity view when selection changes.
        connect(self._clusters_selected, event='select', sender=self.cluster_view)

        # Create the similarity view.
        self.similarity_view = SimilarityView(
            gui, columns=self.columns + ['similarity'], sort=('similarity', 'desc'))
        connect(
            self._get_similar_clusters, event='request_similar_clusters',
            sender=self.similarity_view)
        connect(self._similar_selected, event='select', sender=self.similarity_view)

        # Change the state after every clustering action, according to the action flow.
        connect(self._after_action, event='cluster', sender=self)

    def _reset_cluster_view(self):
        """Recreate the cluster view."""
        logger.debug("Reset the cluster view.")
        self.cluster_view._reset_table(
            data=self.cluster_info, columns=self.columns, sort=self._sort)

    def _clusters_added(self, cluster_ids):
        """Update the cluster and similarity views when new clusters are created."""
        logger.log(5, "Clusters added: %s", cluster_ids)
        data = [self.get_cluster_info(cluster_id) for cluster_id in cluster_ids]
        self.cluster_view.add(data)
        self.similarity_view.add(data)

    def _clusters_removed(self, cluster_ids):
        """Update the cluster and similarity views when clusters are removed."""
        logger.log(5, "Clusters removed: %s", cluster_ids)
        self.cluster_view.remove(cluster_ids)
        self.similarity_view.remove(cluster_ids)

    def _cluster_metadata_changed(self, field, cluster_ids, value):
        """Update the cluster and similarity views when clusters metadata is updated."""
        logger.log(5, "%s changed for %s to %s", field, cluster_ids, value)
        data = [{'id': cluster_id, field: value} for cluster_id in cluster_ids]
        for _ in data:
            _['is_masked'] = _is_group_masked(_.get('group', None))
        self.cluster_view.change(data)
        self.similarity_view.change(data)

    def _clusters_selected(self, sender, obj, **kwargs):
        """When clusters are selected in the cluster view, register the action in the history
        stack, update the similarity view, and emit the global supervisor.select event unless
        update_views is False."""
        if sender != self.cluster_view:
            return
        cluster_ids = obj['selected']
        next_cluster = obj['next']
        kwargs = obj.get('kwargs', {})
        logger.debug("Clusters selected: %s (%s)", cluster_ids, next_cluster)
        self.task_logger.log(self.cluster_view, 'select', cluster_ids, output=obj)
        # Update the similarity view when the cluster view selection changes.
        self.similarity_view.reset(cluster_ids)
        self.similarity_view.set_selected_index_offset(len(self.selected_clusters))
        # Emit supervisor.select event unless update_views is False. This happens after
        # a merge event, where the views should not be updated after the first cluster_view.select
        # event, but instead after the second similarity_view.select event.
        if kwargs.pop('update_views', True):
            emit('select', self, self.selected, **kwargs)
        if cluster_ids:
            self.cluster_view.scroll_to(cluster_ids[-1])
        self.cluster_view.dock.set_status('clusters: %s' % ', '.join(map(str, cluster_ids)))

    def _similar_selected(self, sender, obj):
        """When clusters are selected in the similarity view, register the action in the history
        stack, and emit the global supervisor.select event."""
        if sender != self.similarity_view:
            return
        similar = obj['selected']
        next_similar = obj['next']
        kwargs = obj.get('kwargs', {})
        logger.debug("Similar clusters selected: %s (%s)", similar, next_similar)
        self.task_logger.log(self.similarity_view, 'select', similar, output=obj)
        emit('select', self, self.selected, **kwargs)
        if similar:
            self.similarity_view.scroll_to(similar[-1])
        self.similarity_view.dock.set_status('similar clusters: %s' % ', '.join(map(str, similar)))

    def _on_action(self, sender, name, *args):
        """Called when an action is triggered: enqueue and process the task."""
        assert sender == self.action_creator
        # The GUI should not be busy when calling a new action.
        if 'select' not in name and self._is_busy:
            logger.log(5, "The GUI is busy, waiting before calling the action.")
            try:
                _block(lambda: not self._is_busy)
            except Exception:
                logger.warning("The GUI is busy, could not execute `%s`.", name)
                return
        # Enqueue the requested action.
        self.task_logger.enqueue(self, name, *args)
        # Perform the action (which calls self.<name>(...)).
        self.task_logger.process()

    def _after_action(self, sender, up):
        """Called after an action: update the cluster and similarity views and update
        the selection."""
        # This is called once the action has completed. We update the tables.
        # Update the views with the old and new clusters.
        self._clusters_added(up.added)
        self._clusters_removed(up.deleted)
        self._cluster_metadata_changed(
            up.description.replace('metadata_', ''), up.metadata_changed, up.metadata_value)
        # After the action has finished, we process the pending actions,
        # like selection of new clusters in the tables.
        self.task_logger.process()

    def _set_busy(self, busy):
        # If busy is the same, do nothing.
        if busy is self._is_busy:
            return
        self._is_busy = busy
        # Set the busy cursor.
        logger.log(5, "GUI is %sbusy" % ('' if busy else 'not '))
        set_busy(busy)
        # Let the cluster views know that the GUI is busy.
        self.cluster_view.set_busy(busy)
        self.similarity_view.set_busy(busy)
        # If the GUI is no longer busy, stop the debouncer waiting period.
        if not busy:
            self.cluster_view.debouncer.stop_waiting()

    # Selection actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids, callback=None):
        """Select a list of clusters."""
        # HACK: allow for `select(1, 2, 3)` in addition to `select([1, 2, 3])`
        # This makes it more convenient to select multiple clusters with
        # the snippet: `:c 1 2 3` instead of `:c 1,2,3`.
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
        # Remove non-existing clusters from the selection.
        #cluster_ids = self._keep_existing_clusters(cluster_ids)
        # Update the cluster view selection.
        self.cluster_view.select(cluster_ids, callback=callback)

    # Cluster view actions
    # -------------------------------------------------------------------------

    def sort(self, column, sort_dir='desc'):
        """Sort the cluster view by a given column, in a given order (asc or desc)."""
        self.cluster_view.sort_by(column, sort_dir=sort_dir)

    def filter(self, text):
        """Filter the clusters using a Javascript expression on the column names."""
        self.cluster_view.filter(text)

    def clear_filter(self):
        self.cluster_view.filter('')

    # Properties
    # -------------------------------------------------------------------------

    @property
    def cluster_info(self):
        """The cluster view table as a list of per-cluster dictionaries."""
        return [self.get_cluster_info(cluster_id) for cluster_id in self.clustering.cluster_ids]

    @property
    def shown_cluster_ids(self):
        """The sorted list of cluster ids as they are currently shown in the cluster view."""
        b = Barrier()
        self.cluster_view.get_ids(callback=b(1))
        b.wait()
        return b.result(1)[0][0]

    @property
    def state(self):
        """GUI state, with the cluster view and similarity view states."""
        sc = self.cluster_view.state
        ss = self.similarity_view.state
        return Bunch({'cluster_view': Bunch(sc), 'similarity_view': Bunch(ss)})

    def attach(self, gui):
        """Attach to the GUI."""

        # Make sure the selected field in cluster and similarity views are saved in the local
        # supervisor state, as this information is dataset-dependent.
        gui.state.add_local_keys(['ClusterView.selected'])

        # Create the cluster view and similarity view.
        self._create_views(
            gui=gui, sort=gui.state.get('ClusterView', {}).get('current_sort', None))

        # Create the TaskLogger.
        self.task_logger = TaskLogger(
            cluster_view=self.cluster_view,
            similarity_view=self.similarity_view,
            supervisor=self,
        )

        connect(self._save_gui_state, event='close', sender=gui)
        gui.add_view(self.cluster_view, position='left', closable=False)
        gui.add_view(self.similarity_view, position='left', closable=False)

        # Create all supervisor actions (edit and view menu).
        self.action_creator.attach(gui)
        self.actions = self.action_creator.edit_actions  # clustering actions
        self.select_actions = self.action_creator.select_actions
        self.view_actions = gui.view_actions
        emit('attach_gui', self)

        # Call supervisor.save() when the save/ctrl+s action is triggered in the GUI.
        @connect(sender=gui)
        def on_request_save(sender):
            self.save()

        # Set the debouncer.
        self._busy = {}
        self._is_busy = False
        # Collect all busy events from the views, and sets the GUI as busy
        # if at least one view is busy.

        @connect
        def on_is_busy(sender, is_busy):
            self._busy[sender] = is_busy
            self._set_busy(any(self._busy.values()))

        @connect(sender=gui)
        def on_close(e):
            unconnect(on_is_busy, self)

        @connect(sender=self.cluster_view)
        def on_ready(sender):
            """Select the clusters from the cluster view state."""
            selected = gui.state.get('ClusterView', {}).get('selected', [])
            if selected:  # pragma: no cover
                self.cluster_view.select(selected)

    @property
    def selected_clusters(self):
        """Selected clusters in the cluster view only."""
        state = self.task_logger.last_state()
        return state[0] or [] if state else []

    @property
    def selected_similar(self):
        """Selected clusters in the similarity view only."""
        state = self.task_logger.last_state()
        return state[2] or [] if state else []

    @property
    def selected(self):
        """Selected clusters in the cluster and similarity views."""
        return _uniq(self.selected_clusters + self.selected_similar)

    def n_spikes(self, cluster_id):
        """Number of spikes in a given cluster."""
        return len(self.clustering.spikes_per_cluster.get(cluster_id, []))

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None, to=None):
        """Merge the selected clusters."""
        if cluster_ids is None:
            cluster_ids = self.selected
        if len(cluster_ids or []) <= 1:
            return
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

    # Move actions
    # -------------------------------------------------------------------------

    @property
    def fields(self):
        """List of all cluster label names."""
        return tuple(f for f in self.cluster_meta.fields if f not in ('group',))

    def get_labels(self, field):
        """Return the labels of all clusters, for a given label name."""
        return {c: self.cluster_meta.get(field, c)
                for c in self.clustering.cluster_ids}

    def label(self, name, value, cluster_ids=None):
        """Assign a label to some clusters."""
        if cluster_ids is None:
            cluster_ids = self.selected
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        if len(cluster_ids) == 0:
            return
        self.cluster_meta.set(name, cluster_ids, value)
        self._global_history.action(self.cluster_meta)
        # Add column if needed.
        if name != 'group' and name not in self.columns:
            logger.debug("Add column %s.", name)
            self.columns.append(name)
            self._reset_cluster_view()

    def move(self, group, which):
        """Assign a cluster group to some clusters."""
        if which == 'all':
            which = self.selected
        elif which == 'best':
            which = self.selected_clusters
        elif which == 'similar':
            which = self.selected_similar
        if isinstance(which, int):
            which = [which]
        if not which:
            return
        _ensure_all_ints(which)
        logger.debug("Move %s to %s.", which, group)
        group = 'unsorted' if group is None else group
        self.label('group', group, cluster_ids=which)

    # Wizard actions
    # -------------------------------------------------------------------------

    # There are callbacks because these functions call Javascript functions that return
    # asynchronously in Qt5.

    def reset_wizard(self, callback=None):
        """Reset the wizard."""
        self.cluster_view.first(callback=callback or partial(emit, 'wizard_done', self))

    def next_best(self, callback=None):
        """Select the next best cluster in the cluster view."""
        self.cluster_view.next(callback=callback or partial(emit, 'wizard_done', self))

    def previous_best(self, callback=None):
        """Select the previous best cluster in the cluster view."""
        self.cluster_view.previous(callback=callback or partial(emit, 'wizard_done', self))

    def next(self, callback=None):
        """Select the next cluster in the similarity view."""
        state = self.task_logger.last_state()
        if not state or not state[0]:
            self.cluster_view.first(callback=callback or partial(emit, 'wizard_done', self))
        else:
            self.similarity_view.next(callback=callback or partial(emit, 'wizard_done', self))

    def previous(self, callback=None):
        """Select the previous cluster in the similarity view."""
        self.similarity_view.previous(callback=callback or partial(emit, 'wizard_done', self))

    def unselect_similar(self, callback=None):
        """Select only the clusters in the cluster view."""
        self.cluster_view.select(self.selected_clusters, callback=callback)

    def first(self, callback=None):
        """Select the first cluster in the cluster view."""
        self.cluster_view.first()

    def last(self, callback=None):
        """Select the last cluster in the cluster view."""
        self.cluster_view.last()

    # Other actions
    # -------------------------------------------------------------------------

    def is_dirty(self):
        """Return whether there are any pending changes."""
        return self._is_dirty if self._is_dirty in (False, True) else len(self._global_history) > 1

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

    def block(self):
        """Block until there are no pending actions.

        Only used in the automated testing suite.

        """
        _block(lambda: self.task_logger.has_finished() and not self._is_busy)
        assert not self._is_busy
        _wait(50)
