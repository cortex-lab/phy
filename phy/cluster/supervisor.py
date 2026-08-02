"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import inspect
import logging
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import numpy as np
from phylib.utils import Bunch, connect, emit, unconnect

from phy.gui.actions import Actions
from phy.gui.qt import QAbstractItemView, QHeaderView, Qt, _block, _wait, set_busy
from phy.gui.widgets import Barrier, Table, _uniq

from ._history import GlobalHistory
from ._selection import CurationSelectionController, SelectionChange
from ._utils import create_cluster_meta
from .clustering import Clustering

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
    if l is None or l == []:
        return
    for i in range(len(l)):
        l[i] = int(l[i])


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QueuedTask:
    """One callback-compatible action with its explicit pre-action selection."""

    sender: object
    name: str
    args: tuple
    kwargs: dict
    selection_before: object = None
    next_similar_before: int | None = None

    def __iter__(self):
        # Preserve the long-standing internal four-value task unpacking contract.
        return iter((self.sender, self.name, self.args, self.kwargs))


class TaskLogger:
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
            5,
            'Enqueue %s %s %s %s (%s)',
            sender.__class__.__name__,
            name,
            args,
            kwargs,
            output,
        )
        selection = getattr(self.supervisor, 'selection', None)
        selection_before = selection.snapshot() if selection is not None else None
        next_similar_before = None
        if self.similarity_view is not None and hasattr(self.similarity_view, '_selected_payload'):
            next_similar_before = self.similarity_view._selected_payload()['next']
        self._queue.append(
            QueuedTask(
                sender=sender,
                name=name,
                args=args,
                kwargs=kwargs,
                selection_before=selection_before,
                next_similar_before=next_similar_before,
            )
        )

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
        if hasattr(self.supervisor, '_selection_task_completed'):
            self.supervisor._selection_task_completed(task, output)
        # Find the post tasks after that task has completed, and enqueue them.
        self.enqueue_after(task, output)
        # Loop.
        self.process()

    def _eval(self, task):
        """Evaluate a task and call a callback function."""
        sender, name, args, kwargs = task
        logger.log(5, 'Calling %s.%s(%s)', sender.__class__.__name__, name, args, kwargs)
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
        f = lambda *args, **kwargs: logger.log(5, 'No method _after_%s', name)
        getattr(self, f'_after_{name}', f)(task, output)

    def _after_merge(self, task, output):
        """Tasks that should follow a merge."""
        self.supervisor._select_after_merge(
            output,
            task.selection_before,
            auto_select=self.auto_select_after_action,
            next_similar=task.next_similar_before,
        )

    def _after_split(self, task, output):
        """Tasks that should follow a split."""
        self.supervisor._select_after_split(output)

    def _after_move(self, task, output):
        """Tasks that should follow a move."""
        self.supervisor._select_after_move(task.selection_before, output.metadata_changed)

    def _after_undo(self, task, output):
        """Selection restoration is owned by contextual GlobalHistory entries."""

    def _after_redo(self, task, output):
        """Selection restoration is owned by contextual GlobalHistory entries."""

    def _log(self, task, output):
        """Add a completed task to the history stack."""
        sender, name, args, kwargs = task
        assert sender
        assert name
        logger.log(
            5,
            'Log %s %s %s %s (%s)',
            sender.__class__.__name__,
            name,
            args,
            kwargs,
            output,
        )
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
        for sender, name_, args, kwargs, output in reversed(self._history):
            if (name and name_ == name) or (name_not_in and name_ and name_ not in name_not_in):
                assert name_
                return (sender, name_, args, kwargs, output)

    def show_history(self):
        """Show the history stack."""
        print('=== History ===')
        for sender, name, args, kwargs, output in self._history:
            print(f'{sender.__class__.__name__: <24} {name: <8}', *args, output, kwargs)

    def has_finished(self):
        """Return whether the queue has finished being processed."""
        return len(self._queue) == 0 and not self._processing


# -----------------------------------------------------------------------------
# Cluster view and similarity view
# -----------------------------------------------------------------------------

_CLUSTER_VIEW_STYLES = """
table tr[data-group='good'] {
    color: #86D16D;
}

table tr[data-group='mua'] {
    color: #afafaf;
}

table tr[data-group='noise'] {
    color: #777;
}
"""


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
    skip_masked : bool
        Whether navigation should skip noise and MUA rows.

    """

    _required_columns = ('n_spikes',)
    _view_name = 'cluster_view'
    _styles = _CLUSTER_VIEW_STYLES
    _selection_debounce_delay = 50

    def __init__(
        self,
        *args,
        data=None,
        columns=(),
        sort=None,
        skip_masked=True,
        debounce_delay=None,
    ):
        # NOTE: debounce select events.
        if debounce_delay is None:
            debounce_delay = self._selection_debounce_delay
        Table.__init__(
            self,
            *args,
            title=self.__class__.__name__,
            debounce_events=('select',),
            debounce_delay=debounce_delay,
            skip_masked=skip_masked,
        )
        self._set_styles()
        self._reset_table(data=data, columns=columns, sort=sort)

    def _reset_table(self, data=None, columns=(), sort=None):
        """Recreate the table with specified columns, data, and sort."""
        emit(f'{self._view_name}_init', self)
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

        # Keep group metadata available so the table can style rows based on cluster group.
        value_names = columns + [{'data': ['group']}]
        # Default sort.
        sort = sort or ('n_spikes', 'desc')
        self._init_table(columns=columns, value_names=value_names, data=data, sort=sort)

    def _set_styles(self):
        self.add_style(self._styles)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._similarity_columns_fitted = False

    def set_selected_index_offset(self, n):
        """Set the index of the selected cluster, used for correct coloring in the similarity
        view."""
        Table.set_selected_index_offset(self, n)

    def reset(self, cluster_ids, reference_id=None):
        """Recreate the view for an explicit reference and Cluster-role exclusions."""
        if not len(cluster_ids):
            return
        reference_id = cluster_ids[-1] if reference_id is None else reference_id
        if reference_id not in cluster_ids:
            raise ValueError('The similarity reference must be selected in the Cluster View.')
        similar = emit('request_similar_clusters', self, reference_id)
        # Clear the table.
        if similar:
            rows = [cl for cl in similar[0] if cl['id'] not in cluster_ids]
            fit_columns = bool(rows) and not self._similarity_columns_fitted
            self.remove_all_and_add(rows, fit_columns=fit_columns)
            if fit_columns:
                # The first real similarity payload establishes stable widths for subsequent
                # populated and empty refreshes.
                self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
                self._similarity_columns_fitted = True
        else:  # pragma: no cover
            self.remove_all_and_add([], fit_columns=False)
        return similar


class MergeView(Table):
    """Display the ordered contents of the temporary Merge workspace."""

    def __init__(self, *args, data=None, columns=(), **kwargs):
        super().__init__(*args, title='MERGE MODE', debounce_events=(), **kwargs)
        columns = ['id'] + [column for column in columns if column != 'id']
        self._init_table(
            columns=columns,
            value_names=columns + [{'data': ['group']}],
            data=data,
            sort=None,
        )
        self.filter_edit.hide()
        # A current row is required by QAbstractItemView to initiate a drag. The
        # selection is local interaction state only; every Merge row remains part of
        # the scientific selection projected through ``_selected_ids``.
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)

    def _on_row_clicked(self, index):
        """Rows are workspace members, not an independent selection."""

    def _on_header_clicked(self, section):
        """Merge order changes only through explicit reorder intents."""

    def set_merge_ids(self, cluster_ids, data, color_order):
        """Project one complete ordered Merge session."""
        self.remove_all_and_add(data, fit_columns=not self._column_widths_fitted)
        self.set_selected_index_order(color_order)
        self.set_selected_ids(cluster_ids)

    def _drag_ids_for_index(self, index):
        ids = super()._drag_ids_for_index(index)
        return () if ids and ids[0] == self._reference_id else ids


# -----------------------------------------------------------------------------
# ActionCreator
# -----------------------------------------------------------------------------


class ActionCreator:
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
        # Qt maps Meta to the physical Control key on macOS.
        'select_first_similar': 'meta+space' if sys.platform == 'darwin' else 'ctrl+space',
        'unselect_similar': 'backspace',
        'toggle_merge_mode': 'v',
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
        if not kwargs.get('docstring'):
            kwargs['docstring'] = docstring
        getattr(self, f'{which}_actions').add(emit_fun, name=name, **kwargs)

    def attach(self, gui):
        """Attach the GUI and create the menus."""
        # Create the menus.
        ds = self.default_shortcuts
        dsp = self.default_snippets
        self.edit_actions = Actions(
            gui,
            name='Edit',
            menu='&Edit',
            insert_menu_before='&View',
            default_shortcuts=ds,
            default_snippets=dsp,
        )
        self.select_actions = Actions(
            gui,
            name='Select',
            menu='Sele&ct',
            insert_menu_before='&View',
            default_shortcuts=ds,
            default_snippets=dsp,
        )

        # Create the actions.
        self._create_edit_actions()
        self._create_select_actions()
        self._create_toolbar(gui)

        @connect(sender=gui)
        def on_default_actions_created(sender):
            self._place_merge_actions_before_save(gui)

        self._place_merge_actions_before_save(gui)

    def _create_edit_actions(self):
        w = 'edit'
        self.add(w, 'undo', set_busy=True, icon='f0e2')
        self.add(w, 'redo', set_busy=True, icon='f01e')
        self.edit_actions.separator()

        # Clustering.
        self.add(w, 'merge', set_busy=True, icon='f0c1')
        self.add(w, 'split', set_busy=True)
        self.edit_actions.separator()

        # Move.
        self.add(w, 'move', prompt=True, n_args=2)
        for which in ('best', 'similar', 'all'):
            for group in ('noise', 'mua', 'good', 'unsorted'):
                self.add(
                    w,
                    f'move_{which}_to_{group}',
                    method_name='move',
                    method_args=(group, which),
                    submenu=f'Move {which} to',
                    docstring=f'Move {which} to {group}.',
                )
        self.edit_actions.separator()

        # Label.
        self.add(w, 'label', prompt=True, n_args=2)
        self.edit_actions.separator()

    def _create_select_actions(self):
        w = 'select'

        # Selection.
        self.add(w, 'select', prompt=True, n_args=1)
        self.add(w, 'select_first_similar')
        self.add(
            w,
            'select_n_similar',
            method_name='select_first_similar',
            prompt=True,
            n_args=1,
            prompt_default=lambda: self.supervisor.n_similar_clusters_to_select,
            docstring='Select the first N eligible clusters shown in the similarity view.',
        )
        self.add(w, 'unselect_similar')
        self.add(w, 'toggle_merge_mode', icon='f542')
        self.add(
            w,
            'skip_noise_and_mua',
            method_name='set_skip_masked_clusters',
            checkable=True,
            checked=getattr(self.supervisor, 'skip_masked_clusters', True),
            docstring='Skip noise and MUA clusters during automatic navigation and selection.',
        )
        self.select_actions.get('skip_noise_and_mua').setText('Skip noise and MUA')
        self.select_actions.separator()

        # Sort and filter
        self.add(w, 'filter', prompt=True, n_args=1)
        self.add(w, 'sort', prompt=True, n_args=1)
        self.add(w, 'clear_filter')

        # Sort by:
        for column in getattr(self.supervisor, 'columns', ()):
            self.add(
                w,
                f'sort_by_{column.lower()}',
                method_name='sort',
                method_args=(column,),
                docstring=f'Sort by {column}',
                submenu='Sort by',
                alias=f's{column.replace("_", "")[:2]}',
            )

        self.select_actions.separator()

        # Navigation. Keep traversal commands together rather than splitting
        # the root menu into several small, related sections.
        submenu = 'Navigation'
        self.add(w, 'first', submenu=submenu)
        self.add(w, 'last', submenu=submenu)
        self.select_actions.separator(submenu=submenu)
        self.add(w, 'reset_wizard', icon='f015', submenu=submenu)
        self.select_actions.separator(submenu=submenu)
        self.add(w, 'next', icon='f061', submenu=submenu)
        self.add(w, 'previous', icon='f060', submenu=submenu)
        self.select_actions.separator(submenu=submenu)
        self.add(w, 'next_best', icon='f0a9', submenu=submenu)
        self.add(w, 'previous_best', icon='f0a8', submenu=submenu)

    def _create_toolbar(self, gui):
        gui._toolbar.addAction(self.select_actions.get('reset_wizard'))
        gui._toolbar.addAction(self.select_actions.get('previous_best'))
        gui._toolbar.addAction(self.select_actions.get('next_best'))
        gui._toolbar.addSeparator()
        gui._toolbar.addAction(self.select_actions.get('previous'))
        gui._toolbar.addAction(self.select_actions.get('next'))
        gui._toolbar.addSeparator()
        gui._toolbar.show()

    def _place_merge_actions_before_save(self, gui):
        """Place merge controls beside Save once the default actions exist."""
        save_action = gui.file_actions.get('save')
        if save_action is None:
            return
        toolbar = gui._toolbar
        merge_mode = self.select_actions.get('toggle_merge_mode')
        merge = self.edit_actions.get('merge')
        undo = self.edit_actions.get('undo')
        redo = self.edit_actions.get('redo')
        for action in (
            merge_mode,
            merge,
            undo,
            redo,
            getattr(self, '_merge_history_separator', None),
            getattr(self, '_save_separator', None),
            getattr(self, '_help_separator', None),
        ):
            if action is not None:
                toolbar.removeAction(action)
        toolbar.insertAction(save_action, merge_mode)
        toolbar.insertAction(save_action, merge)
        self._merge_history_separator = toolbar.insertSeparator(save_action)
        toolbar.insertAction(save_action, undo)
        toolbar.insertAction(save_action, redo)
        self._save_separator = toolbar.insertSeparator(save_action)
        help_action = gui.help_actions.get('show_all_shortcuts')
        if help_action is not None:
            self._help_separator = toolbar.insertSeparator(help_action)


# -----------------------------------------------------------------------------
# Clustering GUI component
# -----------------------------------------------------------------------------


def _is_group_masked(group):
    return group in ('noise', 'mua')


class Supervisor:
    """Component that brings manual clustering facilities to a GUI:

    * `Clustering` instance: merge, split, undo, redo.
    * `ClusterMeta` instance: change cluster metadata (e.g. group).
    * Cluster selection.
    * Many manual clustering-related actions, snippets, shortcuts, etc.
    * Two native Qt tables: `ClusterView` and `SimilarityView`.

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
    n_similar_clusters_to_select : int
        Number of rows selected by the select-first-similar action. The default is 15.
    skip_masked_clusters : bool
        Whether automatic navigation and similar-cluster selection skip noise and MUA
        clusters. The default is True.

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

    default_n_similar_clusters_to_select = 15

    def __init__(
        self,
        spike_clusters=None,
        cluster_groups=None,
        cluster_metrics=None,
        cluster_labels=None,
        similarity=None,
        new_cluster_id=None,
        sort=None,
        context=None,
        n_similar_clusters_to_select=None,
        skip_masked_clusters=True,
    ):
        super().__init__()
        self.context = context
        self.similarity = similarity  # function cluster => [(cl, sim), ...]
        self.actions = None  # will be set when attaching the GUI
        self.gui = None
        self.merge_view = None
        self._merge_close_callback = None
        self._merge_dock_state = None
        self._suspend_presentation_order_sync = False
        self._is_dirty = None
        self._sort = sort  # Initial sort requested in the constructor
        # This is populated alongside the existing TaskLogger-derived selection during the
        # migration to an explicit authoritative curation-selection model.
        self.selection = CurationSelectionController()
        self.n_similar_clusters_to_select = self._validate_n_similar_clusters_to_select(
            n_similar_clusters_to_select
            if n_similar_clusters_to_select is not None
            else self.default_n_similar_clusters_to_select
        )
        self.skip_masked_clusters = self._validate_skip_masked_clusters(skip_masked_clusters)

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
            label for label in self.cluster_labels.keys() if label not in self.columns + ['group']
        ]

        # Create Clustering and ClusterMeta.
        # Load the cached spikes_per_cluster array.
        spc = context.load('spikes_per_cluster') if context else None
        self.clustering = Clustering(
            spike_clusters, spikes_per_cluster=spc, new_cluster_id=new_cluster_id
        )

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
        self._global_history = GlobalHistory(
            process_ups=_process_ups,
            restore_context=self._restore_history_context,
        )

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
                    up.descendants, largest_old_cluster=up.largest_old_cluster
                )
            emit('cluster', self, up)

        @connect(sender=self.cluster_meta)  # noqa
        def on_cluster(sender, up):  # noqa
            emit('cluster', self, up)

        connect(self._save_new_cluster_id, event='cluster', sender=self)

        self._is_busy = False

    # Internal methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_n_similar_clusters_to_select(value):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError('n_similar_clusters_to_select must be a positive integer.')
        value = int(value)
        if value <= 0:
            raise ValueError('n_similar_clusters_to_select must be a positive integer.')
        return value

    @staticmethod
    def _validate_skip_masked_clusters(value):
        if not isinstance(value, bool):
            raise ValueError('skip_masked_clusters must be a boolean.')
        return value

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
            logger.info(f'{up.history.title()} cluster assign.')
        elif up.description == 'merge':
            logger.info('Merge clusters %s to %s.', ', '.join(map(str, up.deleted)), up.added[0])
        else:
            logger.info('Assigned %s spikes.', len(up.spike_ids))

    def _log_action_meta(self, sender, up):
        """Log the cluster meta action (move, label)."""
        if sender != self.cluster_meta:
            return
        if up.history:
            logger.info(f'{up.history.title()} move.')
        else:
            logger.info(
                'Change %s for clusters %s to %s.',
                up.description,
                ', '.join(map(str, up.metadata_changed)),
                up.metadata_value,
            )

        # Skip cluster metadata other than groups.
        if up.description != 'metadata_group':
            return

    def _save_new_cluster_id(self, sender, up):
        """Save the new cluster id on disk, knowing that cluster ids are unique for
        easier cache consistency."""
        new_cluster_id = self.clustering.new_cluster_id()
        if self.context:
            logger.log(5, 'Save the new cluster id: %d.', new_cluster_id)
            self.context.save('new_cluster_id', {'new_cluster_id': new_cluster_id})

    def _save_gui_state(self, gui):
        """Save the GUI state with the cluster view and similarity view."""
        if self.selection.state.is_merge_mode:
            self._cancel_merge_mode()
        gui.state.update_view_state(self.cluster_view, self.cluster_view.state)
        gui.state['n_similar_clusters_to_select'] = self.n_similar_clusters_to_select
        gui.state['skip_masked_clusters'] = self.skip_masked_clusters

        # Compatibility no-op on the native table implementation.
        self.cluster_view.clear_temporary_files()
        self.similarity_view.clear_temporary_files()
        if self._merge_close_callback is not None:
            unconnect(self._merge_close_callback)
            self._merge_close_callback = None
        # The GUI is closing and Qt will destroy its native QObject. Do not retain the
        # corresponding Python wrapper until interpreter shutdown.
        self.gui = None

    def _get_similar_clusters(self, sender, cluster_id):
        """Return the clusters similar to a given cluster."""
        sim = self.similarity(cluster_id) or []
        # Only keep existing clusters.
        clusters_set = set(self.clustering.cluster_ids)
        data = [
            dict(similarity=f'{s:.3f}', **self.get_cluster_info(c))
            for c, s in sim
            if c in clusters_set
        ]
        return data

    @staticmethod
    def _table_workflow_state(view):
        """Capture lightweight native-table context used by Merge cancellation."""
        return {
            'sort': tuple(view._current_sort) if view._current_sort else None,
            'filter': view._filter_text,
            'scroll': view.table_view.verticalScrollBar().value(),
        }

    def _workflow_context(self):
        return {
            'cluster': self._table_workflow_state(self.cluster_view),
            'similarity': self._table_workflow_state(self.similarity_view),
        }

    @staticmethod
    def _restore_table_workflow_state(view, state):
        if not state:
            return
        sort = state.get('sort')
        if sort:
            view.sort_by(*sort)
        view.filter(state.get('filter', ''))
        view.table_view.verticalScrollBar().setValue(state.get('scroll', 0))

    def _restore_workflow_context(self, context):
        if not context:
            return
        self._suspend_presentation_order_sync = True
        try:
            self._restore_table_workflow_state(self.cluster_view, context.get('cluster'))
            self._restore_table_workflow_state(self.similarity_view, context.get('similarity'))
        finally:
            self._suspend_presentation_order_sync = False

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
        out['is_masked'] = _is_group_masked(out.get('group'))
        return {k: v for k, v in out.items() if k not in exclude}

    def _create_views(self, gui=None, sort=None):
        """Create the cluster view and similarity view."""

        sort = sort or self._sort  # comes from either the GUI state or constructor

        # Create the cluster view.
        self.cluster_view = ClusterView(
            gui,
            data=self.cluster_info,
            columns=self.columns,
            sort=sort,
            skip_masked=self.skip_masked_clusters,
        )
        # Update the action flow and similarity view when selection changes.
        connect(self._clusters_selected, event='select', sender=self.cluster_view)
        connect(self._table_order_changed, event='table_sort', sender=self.cluster_view)
        connect(self._table_order_changed, event='table_filter', sender=self.cluster_view)

        # Create the similarity view.
        self.similarity_view = SimilarityView(
            gui,
            columns=self.columns + ['similarity'],
            sort=('similarity', 'desc'),
            skip_masked=self.skip_masked_clusters,
        )
        connect(
            self._get_similar_clusters,
            event='request_similar_clusters',
            sender=self.similarity_view,
        )
        connect(self._similar_selected, event='select', sender=self.similarity_view)
        connect(self._table_order_changed, event='table_sort', sender=self.similarity_view)
        connect(self._table_order_changed, event='table_filter', sender=self.similarity_view)
        connect(
            self._add_similar_to_merge_on_right_click,
            event='row_right_click',
            sender=self.similarity_view,
        )

        # Change the state after every clustering action, according to the action flow.
        connect(self._after_action, event='cluster', sender=self)

    def _create_merge_view(self, state=None):
        state = state or self.selection.state
        data = [self.get_cluster_info(cluster_id) for cluster_id in state.merge_ids]
        self.merge_view = MergeView(self.gui, data=data, columns=self.columns)
        self.merge_view._reference_id = state.reference_id
        self.merge_view.configure_cluster_drag_drop(
            'merge', accepted_roles=('merge', 'similarity'), drag_selected_rows=False
        )
        self.similarity_view.configure_cluster_drag_drop(
            'similarity', accepted_roles=('merge',), drag_selected_rows=True
        )
        connect(
            self._remove_merge_candidate_on_right_click,
            event='row_right_click',
            sender=self.merge_view,
        )
        connect(self._on_cluster_drop, event='cluster_drop', sender=self.merge_view)
        connect(self._on_cluster_drop, event='cluster_drop', sender=self.similarity_view)
        self.gui.add_view(self.merge_view, position='left', closable=True)
        if self._merge_dock_state is not None:
            self.gui.restoreState(self._merge_dock_state['window'])
            if self._merge_dock_state['floating']:
                self.merge_view.dock.setFloating(True)
                self.merge_view.dock.restoreGeometry(self._merge_dock_state['geometry'])
        else:
            self.gui.splitDockWidget(self.cluster_view.dock, self.merge_view.dock, Qt.Vertical)
        self.merge_view.dock.setAttribute(Qt.WA_DeleteOnClose)
        self.merge_view.dock.add_button(
            name='cancel_merge_mode',
            text='Cancel Merge Mode',
            callback=lambda checked: self.toggle_merge_mode(),
        )
        return self.merge_view

    def _reset_cluster_view(self):
        """Recreate the cluster view."""
        logger.debug('Reset the cluster view.')
        self.cluster_view._reset_table(
            data=self.cluster_info, columns=self.columns, sort=self._sort
        )

    def _clusters_added(self, cluster_ids):
        """Update the cluster and similarity views when new clusters are created."""
        logger.log(5, 'Clusters added: %s', cluster_ids)
        data = [self.get_cluster_info(cluster_id) for cluster_id in cluster_ids]
        self.cluster_view.add(data)
        self.similarity_view.add(data)

    def _clusters_removed(self, cluster_ids):
        """Update the cluster and similarity views when clusters are removed."""
        logger.log(5, 'Clusters removed: %s', cluster_ids)
        self.cluster_view.remove(cluster_ids)
        self.similarity_view.remove(cluster_ids)

    def _clusters_added_and_removed(self, added, removed):
        """Apply one atomic table mutation for a clustering update."""
        logger.log(5, 'Clusters added: %s; removed: %s', added, removed)
        data = [self.get_cluster_info(cluster_id) for cluster_id in added]
        self.cluster_view.add_remove(data, removed)
        self.similarity_view.add_remove(data, removed)
        if self.merge_view is not None:
            self.merge_view.add_remove(data, removed)

    def _cluster_metadata_changed(self, field, cluster_ids):
        """Update the cluster and similarity views when clusters metadata is updated."""
        data = []
        for cluster_id in cluster_ids:
            group = self.cluster_meta.get('group', cluster_id)
            data.append(
                {
                    'id': cluster_id,
                    field: self.cluster_meta.get(field, cluster_id),
                    'group': group,
                    'is_masked': _is_group_masked(group),
                }
            )
        self.cluster_view.change(data)
        self.similarity_view.change(data)
        if self.merge_view is not None:
            self.merge_view.change(data)

    def _clusters_selected(self, sender, obj, **kwargs):
        """When clusters are selected in the cluster view, register the action in the history
        stack, update the similarity view, and emit the global supervisor.select event unless
        update_views is False."""
        if sender != self.cluster_view:
            return
        if self.selection.state.is_merge_mode:
            logger.warning('Cluster selection is unavailable in Merge mode.')
            return
        if obj.get('revision') not in (None, sender._selection_revision):
            logger.debug('Ignoring stale Cluster View selection revision.')
            return
        cluster_ids = obj['selected']
        next_cluster = obj['next']
        kwargs = obj.get('kwargs', {})
        logger.debug('Clusters selected: %s (%s)', cluster_ids, next_cluster)
        change = self.selection.set_normal_selection(cluster_ids)
        change = self._set_table_presentation_order(change)
        self.task_logger.log(self.cluster_view, 'select', cluster_ids, output=obj)
        # Reset candidates for the newly selected reference without emitting.
        self.similarity_view.reset(cluster_ids, reference_id=change.after.reference_id)
        self.similarity_view.set_selected_ids(())
        self._update_selection_colors()
        self._project_merge_view()
        # Emit supervisor.select event unless update_views is False. This happens after
        # a merge event, where the views should not be updated after the first cluster_view.select
        # event, but instead after the second similarity_view.select event.
        if kwargs.pop('update_views', True):
            emit('select', self, self.selected, **kwargs)
        if cluster_ids:
            self.cluster_view.scroll_to(cluster_ids[-1])
        self.cluster_view.dock.set_status(f'clusters: {", ".join(map(str, cluster_ids))}')

    def _similar_selected(self, sender, obj):
        """When clusters are selected in the similarity view, register the action in the history
        stack, and emit the global supervisor.select event."""
        if sender != self.similarity_view:
            return
        if obj.get('revision') not in (None, sender._selection_revision):
            logger.debug('Ignoring stale Similarity View selection revision.')
            return
        similar = obj['selected']
        next_similar = obj['next']
        kwargs = obj.get('kwargs', {})
        logger.debug('Similar clusters selected: %s (%s)', similar, next_similar)
        presentation_order = self._presentation_order_from_tables(
            self.selection.state, similar_ids=similar
        )
        self.selection.set_similarity_selection(similar, presentation_order)
        self._update_selection_colors()
        self._project_merge_view()
        self.task_logger.log(self.similarity_view, 'select', similar, output=obj)
        emit('select', self, self.selected, **kwargs)
        if similar:
            self.similarity_view.scroll_to(similar[-1])
        self.similarity_view.dock.set_status(f'similar clusters: {", ".join(map(str, similar))}')

    @staticmethod
    def _ids_in_table_order(view, cluster_ids, previous_order=()):
        """Return visible selected IDs followed by hidden IDs in prior presentation order."""
        cluster_ids = tuple(cluster_ids)
        selected = set(cluster_ids)
        visible = [cluster_id for cluster_id in view.get_ids() if cluster_id in selected]
        visible_set = set(visible)
        hidden = [
            cluster_id
            for cluster_id in previous_order
            if cluster_id in selected and cluster_id not in visible_set
        ]
        hidden_set = set(hidden)
        # New role IDs may not yet occur in the previous presentation while a
        # table selection is being applied. They are visible in normal use;
        # retain any exceptional hidden IDs rather than dropping membership.
        hidden.extend(
            cluster_id
            for cluster_id in cluster_ids
            if cluster_id not in visible_set and cluster_id not in hidden_set
        )
        return tuple(visible) + tuple(hidden)

    def _presentation_order_from_tables(self, state, similar_ids=None):
        """Return the active roles in table order without changing their membership."""
        similar_ids = self._ids_in_table_order(
            self.similarity_view,
            state.similar_ids if similar_ids is None else similar_ids,
            state.presentation_order,
        )
        if state.is_merge_mode:
            return state.merge_ids + similar_ids
        cluster_ids = self._ids_in_table_order(
            self.cluster_view, state.cluster_ids, state.presentation_order
        )
        return tuple(
            dict.fromkeys(
                (
                    *((state.reference_id,) if state.reference_id is not None else ()),
                    *cluster_ids,
                    *similar_ids,
                )
            )
        )

    def _set_table_presentation_order(self, change):
        """Apply the current table ordering through the presentation-only transition."""
        presentation_order = self._presentation_order_from_tables(change.after)
        normalized = self.selection.set_presentation_order(presentation_order)
        return SelectionChange.create(change.before, normalized.after)

    def _table_order_changed(self, sender, row_ids):
        """Keep scientific-view order synchronized with selected table rows."""
        if self._suspend_presentation_order_sync:
            return
        if sender is self.cluster_view and self.selection.state.is_merge_mode:
            return
        state = self.selection.state
        change = self._set_table_presentation_order(SelectionChange.create(state, state))
        if not change.presentation_changed:
            return
        self._apply_selection_change(change, refresh_similarity=False)

    def _update_selection_colors(self):
        """Project stable selection-color positions into all workflow tables."""
        state = self.selection.state
        order = state.color_order
        self.cluster_view.set_selected_index_order(order)
        self.similarity_view.set_selected_index_order(order)
        if self.merge_view is not None:
            self.merge_view.set_selected_index_order(order)

    def _project_merge_view(self):
        state = self.selection.state
        if self.merge_view is None or not state.is_merge_mode:
            return
        data = [self.get_cluster_info(cluster_id) for cluster_id in state.merge_ids]
        self.merge_view.set_merge_ids(state.merge_ids, data, state.color_order)
        self.merge_view.dock.set_status(self._merge_status_text())

    def _merge_status_text(self):
        state = self.selection.state
        staged = len(state.merge_ids)
        similar = len(state.similar_ids)
        return f'MERGE MODE — {staged} staged + {similar} selected similar = {staged + similar} clusters'

    def _apply_selection_change(
        self, change, callback=None, refresh_similarity=True, publish=True, sync_presentation=True
    ):
        """Project one complete controller transition and publish it atomically."""
        if sync_presentation:
            change = self._set_table_presentation_order(change)
        state = change.after
        cluster_payload = self.cluster_view.set_selected_ids(state.cluster_ids)
        if refresh_similarity and state.reference_id is not None:
            self.similarity_view.reset(state.merge_ids, reference_id=state.reference_id)
        similar_payload = self.similarity_view.set_selected_ids(state.similar_ids)
        self._update_selection_colors()
        self._project_merge_view()
        self.task_logger.log(
            self.cluster_view,
            'select',
            list(state.cluster_ids),
            output=cluster_payload,
        )
        self.task_logger.log(
            self.similarity_view,
            'select',
            list(state.similar_ids),
            output=similar_payload,
        )
        if publish and change.render_changed:
            emit('select', self, list(state.presentation_order))
        if callback:
            self.cluster_view._schedule_callback(callback, state)

    def _set_merge_mode_ui(self, active):
        self.cluster_view._set_interaction_overlay(
            'CLUSTER VIEW DISABLED IN MERGE MODE\nPress V to re-enable it.' if active else None
        )
        if active:
            self.cluster_view.dock.set_status('MERGE MODE — Cluster View disabled')
        else:
            ids = self.selection.state.cluster_ids
            self.cluster_view.dock.set_status(f'clusters: {", ".join(map(str, ids))}')
        if self.actions is not None:
            can_redo_merge = False
            if active:
                index = self._global_history.current_position + 1
                history = self._global_history._history
                can_redo_merge = index < len(history) and self._is_merge_history_context(
                    history[index].workflow_context
                )
            for name in self.actions._actions_dict:
                enabled = not active or name == 'merge' or (name == 'redo' and can_redo_merge)
                (self.actions.enable if enabled else self.actions.disable)(name)
        if self.select_actions is not None:
            allowed = {
                'toggle_merge_mode',
                'select_first_similar',
                'select_n_similar',
                'unselect_similar',
                'next',
                'previous',
                'skip_noise_and_mua',
            }
            for name in self.select_actions._actions_dict:
                (
                    self.select_actions.enable
                    if not active or name in allowed
                    else self.select_actions.disable
                )(name)

    def _close_merge_view(self):
        view = self.merge_view
        self.merge_view = None
        if view is not None and view in self.gui.views:
            self._merge_dock_state = {
                'window': self.gui.saveState(),
                'floating': view.dock.isFloating(),
                'geometry': view.dock.saveGeometry(),
            }
        if view is not None:
            self._disconnect_merge_view_events(view)
        if view is not None and view in self.gui.views:
            view.dock.close()
        if view is not None:
            unconnect(view.dock)
        self.similarity_view.configure_cluster_drag_drop(None)

    def _disconnect_merge_view_events(self, view):
        """Release event-registry references owned by a temporary Merge View."""
        unconnect(
            view,
            self._on_cluster_drop,
            self._remove_merge_candidate_on_right_click,
        )

    def _on_cluster_drop(self, sender, payload):
        """Translate generic table drops into Merge controller intents."""
        if not self.selection.state.is_merge_mode:
            return
        source = payload['source']
        cluster_ids = payload['cluster_ids']
        insertion = payload['insertion']
        if sender is self.merge_view and source is self.similarity_view:
            insertion = min(max(1, insertion), len(self.selection.state.merge_ids))
            self.add_to_merge(cluster_ids, insertion=insertion)
        elif sender is self.similarity_view and source is self.merge_view:
            self.remove_from_merge(cluster_ids)
        elif sender is self.merge_view and source is self.merge_view:
            current = self.selection.state.merge_ids
            removed_before = sum(
                current.index(cluster_id) < insertion for cluster_id in cluster_ids
            )
            adjusted = max(1, insertion - removed_before)
            self.reorder_merge(cluster_ids, adjusted)

    def _cancel_merge_mode(self, close_view=True):
        if not self.selection.state.is_merge_mode:
            return
        context = self.selection.state.merge.entry_snapshot.workflow_context
        change = self.selection.cancel_merge_mode()
        self._set_merge_mode_ui(False)
        self._apply_selection_change(change, refresh_similarity=False, sync_presentation=False)
        self._restore_workflow_context(context)
        if close_view:
            self._close_merge_view()

    def _restore_history_context(self, selection, workflow_context, direction):
        """Restore a curation snapshot after the associated data undo or redo."""
        if selection.is_merge_mode and self.merge_view is None:
            self._create_merge_view(selection)
            self._set_merge_mode_ui(True)
        elif not selection.is_merge_mode:
            self._set_merge_mode_ui(False)
        change = self.selection.restore(selection)
        self._apply_selection_change(change, refresh_similarity=False, sync_presentation=False)
        if selection.is_merge_mode:
            context = (
                workflow_context.get('tables')
                if self._is_merge_history_context(workflow_context)
                else selection.merge.entry_snapshot.workflow_context
            )
            self._restore_workflow_context(context)
        else:
            self._close_merge_view()

    @staticmethod
    def _is_merge_history_context(context):
        return isinstance(context, dict) and context.get('mode') == 'merge'

    def _select_after_merge(
        self,
        up,
        selection_before,
        *,
        auto_select=False,
        next_similar=None,
    ):
        """Apply the settled post-merge selection from an explicit before snapshot."""
        similar_ids = ()
        if auto_select and selection_before is not None:  # pragma: no cover
            similar_ids = selection_before.similar_ids
            if set(up.deleted).intersection(similar_ids) and next_similar is not None:
                similar_ids = (next_similar,)
            reference_id = up.added[0]
            self.similarity_view.reset((reference_id,), reference_id=reference_id)
            visible = set(self.similarity_view.get_ids())
            similar_ids = tuple(cluster_id for cluster_id in similar_ids if cluster_id in visible)
        change = self.selection.set_normal_selection((up.added[0],), similar_ids)
        self._apply_selection_change(change)

    def _select_after_split(self, up):
        """Select all clusters created by a split as one settled transition."""
        change = self.selection.set_normal_selection(tuple(up.added))
        self._apply_selection_change(change)

    def _select_after_move(self, selection_before, moved_cluster_ids):
        """Apply wizard navigation after metadata changes without task-log reconstruction."""
        if selection_before is None:
            return
        moved = set(moved_cluster_ids)
        cluster_ids = set(selection_before.cluster_ids)
        similar_ids = set(selection_before.similar_ids)

        if moved <= cluster_ids:
            next_clusters = self.cluster_view.selection_after_navigation()
            next_similar = ()
        elif moved <= similar_ids:
            next_clusters = selection_before.cluster_ids
            next_similar = self.similarity_view.selection_after_navigation()
        else:
            next_clusters = self.cluster_view.selection_after_navigation()
            if next_clusters:
                reference_id = next_clusters[0]
                self.similarity_view.reset(next_clusters, reference_id=reference_id)
            next_similar = self.similarity_view.selection_after_navigation()

        change = self.selection.set_normal_selection(next_clusters, next_similar)
        self._apply_selection_change(change)

    def _add_similar_to_merge_on_right_click(self, sender, cluster_id):
        """Transfer a right-clicked Similarity row only into an active Merge workspace."""
        if not self.selection.state.is_merge_mode:
            return
        self.add_to_merge((cluster_id,))

    def _remove_merge_candidate_on_right_click(self, sender, cluster_id):
        emit('action', self.action_creator, 'remove_from_merge', cluster_id)

    def _on_action(self, sender, name, *args):
        """Called when an action is triggered: enqueue and process the task."""
        assert sender == self.action_creator
        if self.selection.state.is_merge_mode and name in {
            'split',
            'label',
            'move',
            'select',
            'sort',
            'filter',
            'clear_filter',
            'first',
            'last',
            'reset_wizard',
            'next_best',
            'previous_best',
            'undo',
        }:
            logger.warning('Action `%s` is unavailable in Merge mode.', name)
            return
        # Ignore wizard navigation requests triggered while another selection task is still
        # being processed. This keeps an explicit select followed immediately by next()
        # from advancing two steps in one block cycle.
        if name == 'next' and self.task_logger._processing:
            return
        # The GUI should not be busy when calling a new action.
        if 'select' not in name and self._is_busy:
            logger.log(5, 'The GUI is busy, waiting before calling the action.')
            try:
                _block(lambda: not self._is_busy)
            except Exception:
                logger.warning('The GUI is busy, could not execute `%s`.', name)
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
        self._clusters_added_and_removed(up.added, up.deleted)
        self._cluster_metadata_changed(
            up.description.replace('metadata_', ''),
            up.metadata_changed,
        )
        if self.selection.state.is_merge_mode:
            self.task_logger.process()
            return
        # Table filtering or cluster removal may make projected rows disappear without a
        # selection event. Keep the authoritative role state synchronized before applying
        # the post-action navigation policy.
        cluster_ids = self.cluster_view.get_selected_ids()
        similar_ids = self.similarity_view.get_selected_ids()
        if tuple(cluster_ids) != self.selection.state.cluster_ids:
            self.selection.set_cluster_selection(cluster_ids)
        if (
            self.selection.state.reference_id is not None
            and tuple(similar_ids) != self.selection.state.similar_ids
        ):
            self.selection.set_similarity_selection(similar_ids)
        # After the action has finished, we process the pending actions,
        # like selection of new clusters in the tables.
        self.task_logger.process()

    def _selection_task_completed(self, task, output):
        """Reconcile navigation results that do not emit a table selection event."""
        sender, name, _, _ = task
        if output is not None or name not in ('next', 'previous'):
            return
        if sender == self.cluster_view:
            self.selection.set_cluster_selection(())
            self.selection.clear_similarity_selection()
        elif sender == self.similarity_view:
            self.selection.clear_similarity_selection()

    def _set_busy(self, busy):
        # If busy is the same, do nothing.
        if busy is self._is_busy:
            return
        self._is_busy = busy
        # Set the busy cursor.
        logger.log(5, f'GUI is {"" if busy else "not "}busy')
        set_busy(busy)
        # Let the cluster views know that the GUI is busy.
        self.cluster_view.set_busy(busy)
        self.similarity_view.set_busy(busy)
        if self.merge_view is not None:
            self.merge_view.set_busy(busy)
        # If the GUI is no longer busy, deliver the latest selection on the next timer tick.
        # Keeping this asynchronous avoids re-entering the task queue during a busy transition.
        if not busy:
            self.cluster_view.debouncer.stop_waiting(delay=0)

    # Selection actions
    # -------------------------------------------------------------------------

    def select(self, *cluster_ids, callback=None):
        """Select a list of clusters."""
        if self.selection.state.is_merge_mode:
            logger.warning('Cluster selection is unavailable in Merge mode.')
            return
        # HACK: allow for `select(1, 2, 3)` in addition to `select([1, 2, 3])`
        # This makes it more convenient to select multiple clusters with
        # the snippet: `:c 1 2 3` instead of `:c 1,2,3`.
        if cluster_ids and isinstance(cluster_ids[0], (tuple, list)):
            cluster_ids = list(cluster_ids[0]) + list(cluster_ids[1:])
        # Remove non-existing clusters from the selection.
        # cluster_ids = self._keep_existing_clusters(cluster_ids)
        # Update the cluster view selection.
        self.cluster_view.select(cluster_ids, callback=callback)

    def _reject_cluster_action_in_merge_mode(self, name):
        if not self.selection.state.is_merge_mode:
            return False
        logger.warning('Action `%s` is unavailable in Merge mode.', name)
        return True

    # Cluster view actions
    # -------------------------------------------------------------------------

    def sort(self, column, sort_dir='desc'):
        """Sort the cluster view by a given column, in a given order (asc or desc)."""
        if self._reject_cluster_action_in_merge_mode('sort'):
            return
        self.cluster_view.sort_by(column, sort_dir=sort_dir)

    def filter(self, text):
        """Filter the clusters using a boolean expression on the column names."""
        if self._reject_cluster_action_in_merge_mode('filter'):
            return
        self.cluster_view.filter(text)

    def clear_filter(self):
        if self._reject_cluster_action_in_merge_mode('clear_filter'):
            return
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

        self.gui = gui

        saved_n_similar = gui.state.get(
            'n_similar_clusters_to_select', self.n_similar_clusters_to_select
        )
        try:
            self.n_similar_clusters_to_select = self._validate_n_similar_clusters_to_select(
                saved_n_similar
            )
        except ValueError:
            logger.warning(
                'Ignoring invalid saved n_similar_clusters_to_select value: %r.',
                saved_n_similar,
            )

        saved_skip_masked = gui.state.get('skip_masked_clusters', self.skip_masked_clusters)
        try:
            self.skip_masked_clusters = self._validate_skip_masked_clusters(saved_skip_masked)
        except ValueError:
            logger.warning(
                'Ignoring invalid saved skip_masked_clusters value: %r.',
                saved_skip_masked,
            )

        # Make sure the selected field in cluster and similarity views are saved in the local
        # supervisor state, as this information is dataset-dependent.
        gui.state.add_local_keys(['ClusterView.selected'])

        # Create the cluster view and similarity view.
        self._create_views(
            gui=gui, sort=gui.state.get('ClusterView', {}).get('current_sort', None)
        )

        # Create the TaskLogger.
        self.task_logger = TaskLogger(
            cluster_view=self.cluster_view,
            similarity_view=self.similarity_view,
            supervisor=self,
        )

        connect(self._save_gui_state, event='close', sender=gui)

        @connect(event='close_view')
        def on_close_view(view, sender):
            if view is self.merge_view:
                self._disconnect_merge_view_events(view)
                unconnect(view.dock)
                self.merge_view = None
                self._cancel_merge_mode(close_view=False)

        self._merge_close_callback = on_close_view

        @connect(sender=self)
        def on_cluster(sender, up):
            self._is_dirty = True
            self._update_save_feedback()

        @connect(sender=gui)
        def on_default_actions_created(sender):
            self._update_save_feedback()

        self._update_save_feedback()

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
        return list(self.selection.state.cluster_ids)

    @property
    def selected_similar(self):
        """Selected clusters in the similarity view only."""
        return list(self.selection.state.similar_ids)

    @property
    def selected_merge(self):
        """Clusters staged in Merge View, or an empty list in Normal mode."""
        state = self.selection.state
        return list(state.merge_ids) if state.is_merge_mode else []

    @property
    def selected(self):
        """Selected clusters in the cluster and similarity views."""
        return list(self.selection.state.presentation_order)

    @property
    def selection_color_order(self):
        """Cluster IDs in their stable selected-color slots."""
        return self.selection.state.color_order

    def n_spikes(self, cluster_id):
        """Number of spikes in a given cluster."""
        return len(self.clustering.spikes_per_cluster.get(cluster_id, []))

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None, to=None):
        """Merge the selected clusters."""
        merge_mode = self.selection.state.is_merge_mode
        if merge_mode and cluster_ids is not None and set(cluster_ids) != set(self.selected):
            logger.warning('An explicit merge cannot differ from the active Merge workspace.')
            return
        if cluster_ids is None:
            cluster_ids = self.selected
        if len(cluster_ids or []) <= 1:
            if merge_mode:
                logger.warning('Select at least one additional candidate before merging.')
            return
        selection_before = self.selection.snapshot()
        workflow_context = (
            {'mode': 'merge', 'tables': self._workflow_context()} if merge_mode else None
        )
        # A merge synchronously emits several related table mutations: metadata
        # inheritance, addition of the merged cluster, and removal of its
        # ancestors. Fit each attached table once after the complete operation
        # instead of rescanning every row after every intermediate mutation.
        with ExitStack() as stack:
            for table_name in ('cluster_view', 'similarity_view', 'merge_view'):
                table = getattr(self, table_name, None)
                if table is not None:
                    stack.enter_context(table.batch_update())
            out = self.clustering.merge(cluster_ids, to=to)
        if not getattr(getattr(self, 'task_logger', None), '_processing', False):
            self._select_after_merge(out, selection_before)
        if merge_mode:
            self._set_merge_mode_ui(False)
            self._close_merge_view()
        self._global_history.action(
            self.clustering,
            description='merge',
            selection_before=selection_before,
            selection_after=self.selection.snapshot(),
            workflow_context=workflow_context,
        )
        return out

    def split(self, spike_ids=None, spike_clusters_rel=0):
        """Make a new cluster out of the specified spikes."""
        if self.selection.state.is_merge_mode:
            logger.warning('Split is unavailable in Merge mode.')
            return
        if spike_ids is None:
            # Concatenate all spike_ids returned by views who respond to request_split.
            spike_ids = emit('request_split', self)
            spike_ids = np.concatenate(spike_ids).astype(np.int64)
            assert spike_ids.dtype == np.int64
            assert spike_ids.ndim == 1
        if len(spike_ids) == 0:
            logger.warning("""No spikes selected, cannot split.""")
            return
        selection_before = self.selection.snapshot()
        task_logger = getattr(self, 'task_logger', None)
        out = self.clustering.split(spike_ids, spike_clusters_rel=spike_clusters_rel)
        if not getattr(task_logger, '_processing', False):
            self._select_after_split(out)
        self._global_history.action(
            self.clustering,
            description='split',
            selection_before=selection_before,
            selection_after=self.selection.snapshot(),
        )
        return out

    # Move actions
    # -------------------------------------------------------------------------

    @property
    def fields(self):
        """List of all cluster label names."""
        return tuple(f for f in self.cluster_meta.fields if f not in ('group',))

    def get_labels(self, field):
        """Return the labels of all clusters, for a given label name."""
        return {c: self.cluster_meta.get(field, c) for c in self.clustering.cluster_ids}

    def label(self, name, value, cluster_ids=None):
        """Assign a label to some clusters."""
        if self.selection.state.is_merge_mode:
            logger.warning('Cluster metadata changes are unavailable in Merge mode.')
            return
        if cluster_ids is None:
            cluster_ids = self.selected
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        if len(cluster_ids) == 0:
            return
        selection_before = self.selection.snapshot()
        self.cluster_meta.set(name, cluster_ids, value)
        self._global_history.action(
            self.cluster_meta,
            description=f'label:{name}',
            selection_before=selection_before,
            selection_after=self.selection.snapshot(),
        )
        # Add column if needed.
        if name != 'group' and name not in self.columns:
            logger.debug('Add column %s.', name)
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
        logger.debug('Move %s to %s.', which, group)
        group = 'unsorted' if group is None else group
        self.label('group', group, cluster_ids=which)

    # Wizard actions
    # -------------------------------------------------------------------------

    # There are callbacks because the table API remains asynchronous for compatibility.

    def reset_wizard(self, callback=None):
        """Reset the wizard."""
        if self._reject_cluster_action_in_merge_mode('reset_wizard'):
            return
        self.cluster_view.first(callback=callback or partial(emit, 'wizard_done', self))

    def next_best(self, callback=None):
        """Select the next best cluster in the cluster view."""
        if self._reject_cluster_action_in_merge_mode('next_best'):
            return
        self.cluster_view.next(callback=callback or partial(emit, 'wizard_done', self))

    def previous_best(self, callback=None):
        """Select the previous best cluster in the cluster view."""
        if self._reject_cluster_action_in_merge_mode('previous_best'):
            return
        self.cluster_view.previous(callback=callback or partial(emit, 'wizard_done', self))

    def next(self, callback=None):
        """Select the next cluster in the similarity view."""
        if self.selection.state.is_merge_mode:
            self.similarity_view.next(callback=callback or partial(emit, 'wizard_done', self))
        elif not self.selected_clusters:
            self.cluster_view.first(callback=callback or partial(emit, 'wizard_done', self))
        else:
            self.similarity_view.next(callback=callback or partial(emit, 'wizard_done', self))

    def previous(self, callback=None):
        """Select the previous cluster in the similarity view."""
        self.similarity_view.previous(callback=callback or partial(emit, 'wizard_done', self))

    def unselect_similar(self, callback=None):
        """Select only the clusters in the cluster view."""
        change = self.selection.clear_similarity_selection()
        self._apply_selection_change(change, callback=callback)

    def toggle_merge_mode(self, callback=None):
        """Enter Merge mode, or cancel the active Merge workspace."""
        if self.selection.state.is_merge_mode:
            self._cancel_merge_mode()
            if callback:
                callback(self.selection.state)
            return self.selection.state
        self.cluster_view.debouncer.flush()
        self.similarity_view.debouncer.flush()
        if not self.selection.state.cluster_ids:
            logger.warning('Select at least one Cluster View row before entering Merge mode.')
            return
        change = self.selection.enter_merge_mode(self._workflow_context())
        self._create_merge_view()
        self._set_merge_mode_ui(True)
        self._apply_selection_change(change, callback=callback)
        return change.after

    def add_to_merge(self, cluster_ids, insertion=None, callback=None):
        """Transfer candidate IDs into the Merge workspace."""
        cluster_ids = tuple(cluster_ids)
        candidates = set(self.similarity_view.get_ids())
        if not set(cluster_ids) <= candidates:
            logger.warning('Merge candidates must be visible in Similarity View.')
            return
        change = self.selection.add_to_merge(cluster_ids, insertion=insertion)
        self._apply_selection_change(change, callback=callback)
        return change.after

    def remove_from_merge(self, cluster_ids, callback=None):
        """Transfer staged candidates back to Similarity View."""
        if isinstance(cluster_ids, Integral):
            cluster_ids = (int(cluster_ids),)
        try:
            change = self.selection.remove_from_merge(cluster_ids)
        except ValueError as e:
            logger.warning('%s', e)
            return
        self._apply_selection_change(change, callback=callback)
        return change.after

    def reorder_merge(self, cluster_ids, insertion, callback=None):
        """Reorder staged candidates and their scientific presentation order."""
        change = self.selection.reorder_merge(cluster_ids, insertion)
        self._apply_selection_change(change, callback=callback)
        return change.after

    def select_first_similar(self, n=None, callback=None):
        """Select N eligible similar clusters, advancing after the current selection."""
        select_from_start = n is not None
        if n is not None:
            self.n_similar_clusters_to_select = self._validate_n_similar_clusters_to_select(n)
        n = self.n_similar_clusters_to_select

        def select(cluster_ids):
            start = 0
            if not select_from_start:
                selected = self.similarity_view.get_selected_ids()
                if selected and selected[-1] in cluster_ids:
                    start = cluster_ids.index(selected[-1]) + 1
            self.similarity_view.select(cluster_ids[start : start + n], callback=callback)

        self.similarity_view.get_navigable_ids(callback=select)

    def set_skip_masked_clusters(self, skip_masked, callback=None):
        """Set whether automatic navigation and selection skip noise and MUA clusters."""
        self.skip_masked_clusters = self._validate_skip_masked_clusters(skip_masked)
        for view_name in ('cluster_view', 'similarity_view'):
            view = getattr(self, view_name, None)
            if view is not None:
                view.skip_masked = self.skip_masked_clusters
        select_actions = getattr(self, 'select_actions', None)
        if select_actions:
            action = select_actions.get('skip_noise_and_mua')
            if action is not None:
                action.setChecked(self.skip_masked_clusters)
        if callback:
            callback(self.skip_masked_clusters)

    def first(self, callback=None):
        """Select the first cluster in the cluster view."""
        if self._reject_cluster_action_in_merge_mode('first'):
            return
        self.cluster_view.first()

    def last(self, callback=None):
        """Select the last cluster in the cluster view."""
        if self._reject_cluster_action_in_merge_mode('last'):
            return
        self.cluster_view.last()

    # Other actions
    # -------------------------------------------------------------------------

    def is_dirty(self):
        """Return whether there are any pending changes."""
        return self._is_dirty if self._is_dirty in (False, True) else len(self._global_history) > 1

    def _update_save_feedback(self, saved=False):
        """Reflect the current curation-save state in the attached GUI."""
        if self.gui is None:
            return
        is_dirty = not saved and self.is_dirty()
        self.gui._set_dirty(is_dirty)
        save_action = self.gui.file_actions.get('save')
        if save_action is not None:
            save_action.setEnabled(is_dirty)
        if saved:
            self.gui.status_message = 'Curation changes saved.'

    def undo(self):
        """Undo the last action."""
        if self.selection.state.is_merge_mode:
            logger.warning('Undo is unavailable while a Merge workspace is active.')
            return
        # Selection-only exploration does not create history entries. Preserve the exact
        # state at the time undo is requested so redo remains a true inverse operation.
        if self._global_history.current_position > 0:
            self._global_history.update_current_context(
                selection_after=self.selection.snapshot(),
            )
        self._global_history.undo()

    def redo(self):
        """Undo the last undone action."""
        if self.selection.state.is_merge_mode:
            index = self._global_history.current_position + 1
            history = self._global_history._history
            if index >= len(history) or not self._is_merge_history_context(
                history[index].workflow_context
            ):
                logger.warning('Redo is unavailable for this Merge workspace.')
                return
        self._global_history.redo()

    def save(self):
        """Save the manual clustering back to disk.

        This method emits the `save_clustering(spike_clusters, groups, *labels)` event.
        It is up to the caller to react to this event and save the data to disk.

        """
        spike_clusters = self.clustering.spike_clusters
        groups = {
            c: self.cluster_meta.get('group', c) or 'unsorted' for c in self.clustering.cluster_ids
        }
        # List of tuples (field_name, dictionary).
        labels = [
            (field, self.get_labels(field))
            for field in self.cluster_meta.fields
            if field not in ('next_cluster')
        ]
        emit('save_clustering', self, spike_clusters, groups, *labels)
        # Cache the spikes_per_cluster array.
        self._save_spikes_per_cluster()
        self._is_dirty = False
        self._update_save_feedback(saved=True)

    def block(self):
        """Block until there are no pending actions.

        Only used in the automated testing suite.

        """
        debouncers = (
            self.cluster_view.debouncer,
            self.similarity_view.debouncer,
        )
        for _ in range(100):
            for debouncer in debouncers:
                debouncer.flush()
            _block(lambda: self.task_logger.has_finished() and not self._is_busy)
            if not any(debouncer.has_pending for debouncer in debouncers):
                break
        else:  # pragma: no cover
            raise RuntimeError('Could not flush pending selections.')
        assert not self._is_busy
        _wait(50)
