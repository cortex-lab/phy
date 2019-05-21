# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import inspect
from itertools import chain
import logging

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering

from phylib.utils import Bunch, emit, connect, unconnect
from phylib.utils._color import ClusterColorSelector
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
    def __init__(self, cluster_view=None, similarity_view=None, supervisor=None):
        self.cluster_view = cluster_view
        self.similarity_view = similarity_view
        self.supervisor = supervisor
        self._processing = False
        # List of tasks that have completed.
        self._history = []
        # Tasks that have yet to be performed.
        self._queue = []

    def enqueue(self, sender, name, *args, output=None):
        logger.log(5, "Enqueue %s %s %s (%s)", sender.__class__.__name__, name, args, output)
        self._queue.append((sender, name, args))

    def dequeue(self):
        return self._queue.pop(0) if self._queue else None

    def _callback(self, task, output):
        # Log the task and its output.
        self._log(task, output)
        # Find the post tasks after that task has completed, and enqueue them.
        self.enqueue_after(task, output)
        # Loop.
        self.process()

    def _eval(self, task):
        # Evaluation a task and call a callback function.
        sender, name, args = task
        logger.log(5, "Calling %s.%s(%s)", sender.__class__.__name__, name, args)
        f = getattr(sender, name)
        callback = partial(self._callback, task)
        argspec = inspect.getfullargspec(f)
        argspec = argspec.args + argspec.kwonlyargs
        if 'callback' in argspec:
            f(*args, callback=callback)
        else:
            # HACK: use on_cluster event instead of callback.
            def _cluster_callback(tsender, up):
                self._callback(task, up)
            connect(_cluster_callback, event='cluster', sender=self.supervisor)
            f(*args)
            unconnect(_cluster_callback)

    def process(self):
        self._processing = True
        task = self.dequeue()
        if not task:
            self._processing = False
            return
        # Process the first task in queue, or stop if the queue is empty.
        self._eval(task)

    def enqueue_after(self, task, output):
        sender, name, args = task
        getattr(self, '_after_%s' % name,
                lambda *args: logger.log(5, "No method _after_%s", name))(task, output)

    def _after_merge(self, task, output):
        sender, name, args = task
        merged, to = output.deleted, output.added[0]
        self.enqueue(self.cluster_view, 'select', [to])
        cluster_ids, next_cluster, similar, next_similar = self.last_state()
        if similar is None:
            return
        if set(merged).intersection(similar) and next_similar is not None:
            self.enqueue(self.similarity_view, 'select', [next_similar])
        else:
            self.enqueue(self.similarity_view, 'select', similar)

    def _after_split(self, task, output):
        sender, name, args = task
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
        sender, name, args = task
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
        last_action = self.last_task(name_not_in=('select', 'next', 'previous', 'undo', 'redo'))
        self._select_state(self.last_state(last_action))

    def _after_redo(self, task, output):
        last_undo = self.last_task('undo')
        # Select the last state before the last undo.
        self._select_state(self.last_state(last_undo))

    def _select_state(self, state):
        cluster_ids, next_cluster, similar, next_similar = state
        self.enqueue(self.cluster_view, 'select', cluster_ids)
        if similar:
            self.enqueue(self.similarity_view, 'select', similar)

    def _log(self, task, output):
        sender, name, args = task
        assert sender
        assert name
        logger.log(5, "Log %s %s %s (%s)", sender.__class__.__name__, name, args, output)
        task = (sender, name, args, output)
        # Avoid successive duplicates (even if sender is different).
        if not self._history or self._history[-1][1:] != task[1:]:
            self._history.append(task)

    def log(self, sender, name, *args, output=None):
        self._log((sender, name, args), output)

    def last_task(self, name=None, name_not_in=()):
        for (sender, name_, args, output) in reversed(self._history):
            if (name and name_ == name) or (name_not_in and name_ and name_ not in name_not_in):
                assert name_
                return (sender, name_, args, output)

    def last_state(self, task=None):
        """Return (cluster_ids, next_cluster, similar, next_similar)."""
        cluster_state = (None, None)
        similarity_state = (None, None)
        h = self._history
        # Last state until the passed task, if applicable.
        if task:
            i = self._history.index(task)
            h = self._history[:i]
        for (sender, name, args, output) in reversed(h):
            # Last selection is cluster view selection: return the state.
            if (sender == self.similarity_view and
                    similarity_state == (None, None) and
                    name in ('select', 'next', 'previous')):
                similarity_state = output or (None, None)
            if (sender == self.cluster_view and
                    cluster_state == (None, None) and
                    name in ('select', 'next', 'previous')):
                cluster_state = output or (None, None)
                return (*cluster_state, *similarity_state)

    def show_history(self):
        print("=== History ===")
        for sender, name, args, output in self._history:
            print('{: <24} {: <8}'.format(sender.__class__.__name__, name), *args, output)

    def has_finished(self):
        return len(self._queue) == 0 and not self._processing


# -----------------------------------------------------------------------------
# Cluster view and similarity view
# -----------------------------------------------------------------------------

class ClusterView(Table):
    _required_columns = ('n_spikes',)
    _view_name = 'cluster_view'

    def __init__(self, *args, data=None, columns=(), sort=None):
        HTMLWidget.__init__(self, *args, title=self.__class__.__name__)
        self._set_styles()
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
        self.builder.add_style('''
            table tr[data-group='good'] {
                color: #86D16D;
            }

            table tr[data-group='mua'], table tr[data-group='noise'] {
                color: #888;
            }
            ''')

    def get_state(self, callback=None):
        self.get_current_sort(lambda sort: callback({'current_sort': tuple(sort or (None, None))}))

    def set_state(self, state):
        sort_by, sort_dir = state.get('current_sort', (None, None))
        if sort_by:
            self.sort_by(sort_by, sort_dir)


class SimilarityView(ClusterView):
    _required_columns = ('n_spikes', 'similarity')
    _view_name = 'similarity_view'

    def set_selected_index_offset(self, n):
        self.eval_js('table._setSelectedIndexOffset(%d);' % n)

    def reset(self, cluster_ids):
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
        # This special keyword argument lets us use a different name for the
        # action and the event name/method (used for different move flavors).
        method_name = kwargs.pop('method_name', name)
        method_args = kwargs.pop('method_args', ())
        emit_fun = partial(emit, 'action', self, method_name, *method_args)
        self.actions.add(emit_fun, name=name, **kwargs)

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
        self.add('move', prompt=True, n_args=2,
                 docstring='Move some clusters to a group.')
        self.separator()

        for which in ('best', 'similar', 'all'):
            for group in ('noise', 'mua', 'good', 'unsorted'):
                self.add('move_%s_to_%s' % (which, group),
                         method_name='move',
                         method_args=(group, which),
                         docstring='Move %s to %s.' % (which, group))
            self.separator()

        # Label.
        self.add('label', alias='l', prompt=True, n_args=2,
                 docstring='Label the selected clusters.')

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

def _is_group_masked(group):
    return group in ('noise', 'mua')


class Supervisor(object):
    """Component that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Parameters
    ----------

    spike_clusters : ndarray
    cluster_groups : dictionary {cluster_id: group_name}
    cluster_metrics : dictionary {metrics_name: func cluster_id => value}
    similarity: func
    new_cluster_id: func
    context: Context instance

    GUI events
    ----------

    When this component is attached to a GUI, the following events are emitted:

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
                 cluster_metrics=None,
                 cluster_labels=None,
                 similarity=None,
                 new_cluster_id=None,
                 sort=None,
                 context=None,
                 ):
        super(Supervisor, self).__init__()
        self.context = context
        self.similarity = similarity  # function cluster => [(cl, sim), ...]

        self._init_sort = sort

        # Cluster metrics.
        # This is a dict {name: func cluster_id => value}.
        self.cluster_metrics = cluster_metrics or {}
        self.cluster_metrics['n_spikes'] = self.n_spikes

        # Cluster labels.
        # This is a dict {name: {cl: value}}
        self.cluster_labels = cluster_labels or {}

        self.columns = ['id']  # n_spikes comes from cluster_metrics
        self.columns += [label for label in self.cluster_labels.keys() if label != 'group']
        self.columns += list(self.cluster_metrics.keys())

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
        self.action_creator = ActionCreator()
        connect(self._on_action, event='action', sender=self.action_creator)

        # Log the actions.
        connect(self._log_action, event='cluster', sender=self.clustering)
        connect(self._log_action_meta, event='cluster', sender=self.cluster_meta)

        # Raise supervisor.cluster
        @connect(sender=self.clustering)
        def on_cluster(sender, up):
            emit('cluster', self, up)

        @connect(sender=self.cluster_meta)  # noqa
        def on_cluster(sender, up):
            emit('cluster', self, up)

        connect(self._save_new_cluster_id, event='cluster', sender=self)

        self._is_busy = False

    # Internal methods
    # -------------------------------------------------------------------------

    def _set_color_actions(self):
        # Create the ClusterColorSelector instance.
        self.color_selector = ClusterColorSelector(
            cluster_meta=self.cluster_meta,
            cluster_metrics=self.cluster_metrics,
            cluster_ids=self.clustering.cluster_ids,
        )

        # Change color field action.
        def _make_color_field_action(color_field):
            def change_color_field():
                self.color_selector.set_color_mapping(field=color_field)
                emit('color_mapping_changed', self)
            return change_color_field

        for field in chain(
                ('cluster', 'group', 'n_spikes'),
                self.cluster_labels.keys(), self.cluster_metrics.keys()):
            self.actions.add(
                _make_color_field_action(field), name='Color field: %s' % field.lower(),
                menu='Co&lor', submenu='Change color field')

        # Change color map action.
        def _make_colormap_action(colormap):
            def change_colormap():
                self.color_selector.set_color_mapping(colormap=colormap)
                emit('color_mapping_changed', self)
            return change_colormap

        for colormap in ('categorical', 'linear', 'diverging', 'rainbow'):
            self.actions.add(
                _make_colormap_action(colormap), name='Colormap: %s' % colormap.lower(),
                menu='Co&lor', submenu='Change colormap')

        # Change colormap categorical or continous.
        @self.actions.add(menu='Co&lor', checkable=True, checked=True)
        def toggle_categorical(checked):
            self.color_selector.set_color_mapping(categorical=checked)
            emit('color_mapping_changed', self)

        @connect(sender=self)
        def on_cluster(sender, up):
            # After a clustering action, get the cluster ids as shown
            # in the cluster view, and update the color selector accordingly.
            @self.cluster_view.get_ids
            def _update(cluster_ids):
                self.color_selector.set_cluster_ids(cluster_ids)
                emit('color_mapping_changed', self)

    def _save_spikes_per_cluster(self):
        if not self.context:
            return
        self.context.save('spikes_per_cluster',
                          self.clustering.spikes_per_cluster,
                          kind='pickle',
                          )

    def _log_action(self, sender, up):
        if sender != self.clustering:
            return
        if up.history:
            logger.info(up.history.title() + " cluster assign.")
        elif up.description == 'merge':
            logger.info("Merge clusters %s to %s.",
                        ', '.join(map(str, up.deleted)),
                        up.added[0])
        else:
            logger.info("Assigned %s spikes.", len(up.spike_ids))

    def _log_action_meta(self, sender, up):
        if sender != self.cluster_meta:
            return
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

    def _save_new_cluster_id(self, sender, up):
        # Save the new cluster id on disk.
        new_cluster_id = self.clustering.new_cluster_id()
        if self.context:
            logger.debug("Save the new cluster id: %d.", new_cluster_id)
            self.context.save('new_cluster_id',
                              dict(new_cluster_id=new_cluster_id))

    def _save_gui_state(self, gui):
        b = Barrier()
        self.cluster_view.get_state(b(1))
        b.wait()
        state = b.result(1)[0][0]
        gui.state.update_view_state(self.cluster_view, state)

    def n_spikes(self, cluster_id):
        return len(self.clustering.spikes_per_cluster.get(cluster_id, []))

    def _get_similar_clusters(self, sender, cluster_id):
        sim = self.similarity(cluster_id)
        # Only keep existing clusters.
        clusters_set = set(self.clustering.cluster_ids)
        data = [dict(similarity='%.3f' % s,
                     **self._get_cluster_info(c))
                for c, s in sim
                if c in clusters_set]
        return data

    def _get_cluster_info(self, cluster_id, exclude=()):
        out = {'id': cluster_id,
               # 'n_spikes': self.n_spikes(cluster_id),
               }
        for key, func in self.cluster_metrics.items():
            out[key] = func(cluster_id)
        for key in self.cluster_meta.fields:
            # includes group
            out[key] = self.cluster_meta.get(key, cluster_id)
        out['is_masked'] = _is_group_masked(out.get('group', None))
        return {k: v for k, v in out.items() if k not in exclude}

    def _create_views(self, gui=None):
        data = [self._get_cluster_info(cluster_id) for cluster_id in self.clustering.cluster_ids]

        # Create the cluster view.
        self.cluster_view = ClusterView(
            gui, data=data, columns=self.columns, sort=self._init_sort)
        # Update the action flow and similarity view when selection changes.
        connect(self._clusters_selected, event='select', sender=self.cluster_view)

        # Create the similarity view.
        self.similarity_view = SimilarityView(
            gui, columns=self.columns + ['similarity'], sort=('similarity', 'desc'))
        connect(self._get_similar_clusters, event='request_similar_clusters',
                sender=self.similarity_view)
        connect(self._similar_selected, event='select', sender=self.similarity_view)

        # Change the state after every clustering action, according to the action flow.
        connect(self._after_action, event='cluster', sender=self)

    def _clusters_added(self, cluster_ids):
        logger.log(5, "Clusters added: %s", cluster_ids)
        data = [self._get_cluster_info(cluster_id) for cluster_id in cluster_ids]
        self.cluster_view.add(data)
        self.similarity_view.add(data)

    def _clusters_removed(self, cluster_ids):
        logger.log(5, "Clusters removed: %s", cluster_ids)
        self.cluster_view.remove(cluster_ids)
        self.similarity_view.remove(cluster_ids)

    def _cluster_metadata_changed(self, field, cluster_ids, value):
        logger.log(5, "%s changed for %s to %s", field, cluster_ids, value)
        data = [{'id': cluster_id,
                 field: value,  # self.cluster_meta.get(field, cluster_id),
                 }
                for cluster_id in cluster_ids]
        for _ in data:
            _['is_masked'] = _is_group_masked(_.get('group', None))
        self.cluster_view.change(data)
        self.similarity_view.change(data)

    def _clusters_selected(self, sender, cluster_ids_and_next):
        if sender != self.cluster_view:
            return
        cluster_ids, next_cluster = cluster_ids_and_next
        logger.debug("Clusters selected: %s (%s)", cluster_ids, next_cluster)
        self.task_logger.log(self.cluster_view, 'select', cluster_ids, output=cluster_ids_and_next)
        # Update the similarity view when the cluster view selection changes.
        self.similarity_view.reset(cluster_ids)
        self.similarity_view.set_selected_index_offset(len(self.selected_clusters))
        emit('select', self, self.selected)

    def _similar_selected(self, sender, similar_and_next):
        if sender != self.similarity_view:
            return
        similar, next_similar = similar_and_next
        logger.debug("Similar clusters selected: %s (%s)", similar, next_similar)
        self.task_logger.log(self.similarity_view, 'select', similar, output=similar_and_next)
        emit('select', self, self.selected)

    def _on_action(self, sender, name, *args):
        """Bind the 'action' event raised by ActionCreator to methods of this class."""
        if sender != self.action_creator:
            return
        # The GUI should not be busy when calling a new action.
        if 'select' not in name and self._is_busy:
            logger.log(5, "The GUI is busy, waiting before calling the action.")
            _block(lambda: not self._is_busy)
        # Enqueue the requested action.
        self.task_logger.enqueue(self, name, *args)
        # Perform the action (which calls self.<name>(...)).
        self.task_logger.process()

    def _after_action(self, sender, up):
        # This is called once the action has completed. We update the tables.
        # Update the views with the old and new clusters.
        self._clusters_added(up.added)
        self._clusters_removed(up.deleted)
        self._cluster_metadata_changed(
            up.description.replace('metadata_', ''), up.metadata_changed, up.metadata_value)
        # After the action has finished, we process the pending actions,
        # like selection of new clusters in the tables.
        self.task_logger.process()

    @property
    def state(self):
        b = Barrier()
        self.cluster_view.get_state(b(1))
        self.similarity_view.get_state(b(2))
        b.wait()
        sc = b.result(1)[0][0]
        ss = b.result(2)[0][0]
        return Bunch({'cluster_view': Bunch(sc), 'similarity_view': Bunch(ss)})

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

    def attach(self, gui):
        # Create the cluster view and similarity view.
        self._create_views(gui=gui)

        # Create the TaskLogger.
        self.task_logger = TaskLogger(
            cluster_view=self.cluster_view,
            similarity_view=self.similarity_view,
            supervisor=self,
        )

        connect(self._save_gui_state, event='close', sender=gui)
        gui.add_view(self.cluster_view, position='left')
        gui.add_view(self.similarity_view, position='left')
        #self.cluster_view.set_state(gui.state.get_view_state(self.cluster_view, gui))

        self.action_creator.attach(gui)

        # Create the cluster color selector and associated actions.
        self._set_color_actions()

        emit('attach_gui', self)

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
            unconnect(on_is_busy)

    @property
    def actions(self):
        """Works only after a GUI has been attached to the supervisor."""
        return getattr(self.action_creator, 'actions', None)

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

    # Clustering actions
    # -------------------------------------------------------------------------

    @property
    def selected_clusters(self):
        state = self.task_logger.last_state()
        return state[0] or [] if state else []

    @property
    def selected_similar(self):
        state = self.task_logger.last_state()
        return state[2] or [] if state else []

    @property
    def selected(self):
        return _uniq(self.selected_clusters + self.selected_similar)

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
        """Split the selected spikes."""
        if spike_ids is None:
            spike_ids = emit('request_split', self, single=True)
            spike_ids = np.asarray(spike_ids, dtype=np.int64)
            assert spike_ids.dtype == np.int64
            assert spike_ids.ndim == 1
        if len(spike_ids) == 0:
            msg = ("You first need to select spikes in the feature "
                   "view with a few Ctrl+Click around the spikes "
                   "that you want to split.")
            emit('error', self, msg)
            return
        out = self.clustering.split(
            spike_ids, spike_clusters_rel=spike_clusters_rel)
        self._global_history.action(self.clustering)
        return out

    # Move actions
    # -------------------------------------------------------------------------

    @property
    def fields(self):
        """Tuple of label fields."""
        return tuple(f for f in self.cluster_meta.fields
                     if f not in ('group',))

    def get_labels(self, field):
        """Return the labels of all clusters, for a given field."""
        return {c: self.cluster_meta.get(field, c)
                for c in self.clustering.cluster_ids}

    def label(self, name, value, cluster_ids=None):
        """Assign a label to clusters."""
        if cluster_ids is None:
            cluster_ids = self.selected
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        if len(cluster_ids) == 0:
            return
        self.cluster_meta.set(name, cluster_ids, value)
        self._global_history.action(self.cluster_meta)

    def move(self, group, which):
        """Assign a group to some clusters."""
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

    def reset(self, callback=None):
        """Reset the wizard."""
        self.cluster_view.first(callback=callback or partial(emit, 'wizard_done', self))

    def next_best(self, callback=None):
        """Select the next best cluster."""
        self.cluster_view.next(callback=callback or partial(emit, 'wizard_done', self))

    def previous_best(self, callback=None):
        """Select the previous best cluster."""
        self.cluster_view.previous(callback=callback or partial(emit, 'wizard_done', self))

    def next(self, callback=None):
        """Select the next cluster."""
        state = self.task_logger.last_state()
        if not state or not state[0]:
            self.cluster_view.first(callback=callback or partial(emit, 'wizard_done', self))
        else:
            self.similarity_view.next(callback=callback or partial(emit, 'wizard_done', self))

    def previous(self, callback=None):
        """Select the previous cluster."""
        self.similarity_view.previous(callback=callback or partial(emit, 'wizard_done', self))

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
        emit('request_save', self, spike_clusters, groups, *labels)
        # Cache the spikes_per_cluster array.
        self._save_spikes_per_cluster()

    def block(self):
        """Block until there are no pending actions."""
        _block(lambda: self.task_logger.has_finished() and not self._is_busy)
        assert not self._is_busy
        # self.task_logger.show_history()
        _wait(50)
