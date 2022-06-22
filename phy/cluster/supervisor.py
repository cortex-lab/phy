# -*- coding: utf-8 -*-

"""Supervisor."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import inspect
import logging
from pprint import pprint

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering

from phylib.utils import Bunch, emit, connect, unconnect, silent
from phy.gui.actions import Actions
from phy.gui.gui import GUI
from phy.gui.qt import _block, set_busy, _wait
from phy.gui.widgets import Table, _uniq

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _is_group_masked(group):
    return group in ('noise', 'mua')


# -----------------------------------------------------------------------------
# Cluster view and similarity view
# -----------------------------------------------------------------------------

GROUP_COLORS = {
    'good': '#86D16D',
    'mua': '#afafaf',
    'noise': '#777',
}


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

    def __init__(self, *args, data=None, columns=(), sort=None):
        Table.__init__(self, *args, title=self.__class__.__name__)
        self._reset_table(data=data, columns=columns, sort=sort)

    def _set_item_style(self, row_idx, col_idx, d):
        """Set row color as a function of a cluster's group."""
        # mask = d.get('is_masked', False)
        group = d.get('group', None)
        if group:
            d['_foreground'] = GROUP_COLORS.get(group, None)
        super(ClusterView, self)._set_item_style(row_idx, col_idx, d)

    def _reset_table(self, data=None, columns=(), sort=None):
        """Recreate the table with specified columns, data, and sort."""
        emit(self._view_name + '_init', self)

        # Ensure 'id' is the first column.
        columns = ['id'] + [_ for _ in columns if _ != 'id']

        # Add required columns if needed.
        for col in self._required_columns:
            if col not in columns:
                columns += [col]
            assert col in columns
        assert columns[0] == 'id'

        # Default sort.
        sort = sort or ('n_spikes', 'desc')
        self._init_table(columns=columns, data=data, sort=sort)

    @property
    def state(self):
        """Return the cluster view state, with the current sort and selection."""
        current_sort = self.get_current_sort()
        selected = self.get_selected()

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

    """

    _required_columns = ('n_spikes', 'similarity')
    _view_name = 'similarity_view'
    _selected_index_offset = 0

    def set_selected_index_offset(self, n):
        """Set the index of the selected cluster, used for correct coloring in the similarity
        view."""
        self._selected_index_offset = n


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
        self.callbacks = {}

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

        def raise_action():
            # NOTE: only 1 callback per action is supported for now.
            logger.log(5, f"raising action {method_name}{method_args}")
            f = self.callbacks.get(method_name, None)
            if f:
                return f(*method_args)

        getattr(self, '%s_actions' % which).add(raise_action, name=name, **kwargs)

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

    def connect(self, f):
        """Register a callback for a transition, defined by the function's name e.g. on_merge."""
        name = f.__name__
        if not name.startswith('on_'):
            raise ValueError(f"function name `{f}` should start with on_")
        self.callbacks[name[3:]] = f

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
# TableController
# -----------------------------------------------------------------------------

class TableController:
    """Create the cluster view and similarity view."""

    default_similarity_sort = ('similarity', 'desc')
    default_cluster_sort = ('n_spikes', 'desc')

    def __init__(
        self, gui,
        cluster_ids=None,  # [int]
        cluster_groups=None,  # {int: str}
        cluster_labels=None,  # {str: {int: float}}
        cluster_metrics=None,  # {str: int => float}
        similarity=None,  # [int] => [tuple(int, float)]
        sort=None,  # (str, 'asc'|'desc')
        columns=None,  # [str]
    ):
        # create cluster view and similarity view
        # define the next, prev functions etc to pass to the Automaton
        assert gui
        assert isinstance(gui, GUI)
        self.gui = gui

        self.cluster_groups = cluster_groups or {}
        self.cluster_ids = \
            list(cluster_ids if cluster_ids is not None else sorted(self.cluster_groups.keys()))
        self.cluster_labels = cluster_labels or {}
        self.cluster_metrics = cluster_metrics or {}
        self.fn_similarity = similarity
        self.sort = sort or self.default_cluster_sort
        self.columns = columns or self._default_columns()

        # Create tables.
        self._create_cluster_view()
        self._create_similarity_view()

    # Private methods
    # -------------------------------------------------------------------------

    def _default_columns(self):
        columns = ['id']
        columns += list(self.cluster_metrics.keys())
        columns += [
            label for label in self.cluster_labels.keys()
            if label not in columns  # + ['group']
        ]
        return columns

    def _create_cluster_view(self):
        """Create the cluster view."""

        # Create the cluster view.
        self.cluster_view = ClusterView(
            self.gui,
            data=self.cluster_info(),
            columns=self.columns,
            sort=self.sort,
        )
        self.gui.add_view(self.cluster_view, position='left', closable=False)

        connect(self._clusters_selected, event='select', sender=self.cluster_view)

    def _get_data_similar(self, selected_clusters):
        """Return the data to feed the similarity view."""
        sim = self.fn_similarity(selected_clusters) or []
        # Only keep existing clusters.
        all_clusters = set(self.cluster_ids)
        data = [
            dict(similarity=s, **self.get_cluster_info(c)) for c, s in sim if c in all_clusters]
        return data

    def _create_similarity_view(self):
        """Create the similarity view."""

        # Create the similarity view.
        self.similarity_view = SimilarityView(
            self.gui,
            columns=self.columns + ['similarity'],
            sort=self.default_similarity_sort,
        )
        self.gui.add_view(self.similarity_view, position='right', closable=False)

    # Cluster info
    # -------------------------------------------------------------------------

    def get_cluster_info(self, cluster_id, exclude=()):
        """Return the data associated to a given cluster."""
        out = {'id': cluster_id}

        # Cluster metrics.
        for key, func in self.cluster_metrics.items():
            out[key] = func(cluster_id)

        # Cluster labels.
        for key, d in self.cluster_labels.items():
            out[key] = d.get(cluster_id, None)

        # Cluster groups.
        out['group'] = self.cluster_groups.get(cluster_id, None)
        out['is_masked'] = _is_group_masked(out.get('group', None))

        # Exclude some fields.
        for ex in exclude:
            del out[ex]

        return out

    def cluster_info(self):
        """The cluster info as a list of per-cluster dictionaries."""
        return [self.get_cluster_info(cluster_id) for cluster_id in self.cluster_ids]

    def reset_similarity_view(self, selected_clusters):
        """Reset the similarity view with the clusters similar to the specified clusters."""
        self.similarity_view.remove_all_and_add(self._get_data_similar(selected_clusters))

    # Properties
    # -------------------------------------------------------------------------

    @property
    def shown_cluster_ids(self):
        """The sorted list of cluster ids as they are currently shown in the cluster view."""
        return self.cluster_view.shown_ids()

    @property
    def selected_clusters(self):
        """Selected clusters in the cluster view only."""
        return self.cluster_view.get_selected()

    @property
    def selected_similar(self):
        """Selected clusters in the similarity view only."""
        return self.similarity_view.get_selected()

    @property
    def selected(self):
        """Selected clusters in the cluster and similarity views."""
        return _uniq(self.selected_clusters + self.selected_similar)

    # Event callbacks
    # -------------------------------------------------------------------------

    def _clusters_selected(self, sender, obj, **kwargs):
        """When clusters are selected in the cluster view, register the action in the history
        stack, update the similarity view, and emit the global supervisor.select event unless
        update_views is False."""
        if sender != self.cluster_view:
            return
        selected = obj['selected']
        next_cluster = obj['next']
        kwargs = obj.get('kwargs', {})
        logger.debug(f"Clusters selected: {selected} ({next_cluster})")

        # Update the similarity view when the cluster view selection changes.
        self.similarity_view.set_selected_index_offset(len(self.selected_clusters))
        self.reset_similarity_view(selected)

        # Emit supervisor.select event unless update_views is False. This happens after
        # a merge event, where the views should not be updated after the first cluster_view.select
        # event, but instead after the second similarity_view.select event.
        # if kwargs.pop('update_views', True):
        #     emit('select', self, self.selected, **kwargs)
        if selected:
            self.cluster_view.scroll_to(selected[-1])
        self.cluster_view.dock.set_status('clusters: %s' % ', '.join(map(str, selected)))

    # Selection methods
    # -------------------------------------------------------------------------

    def select_clusters(self, cluster_ids):
        """Select clusters in the cluster view."""
        self.cluster_view.select(cluster_ids)

    def select_similar(self, cluster_ids):
        """Select clusters in the similarity view."""
        self.similarity_view.select(cluster_ids)

    # Update methods
    # -------------------------------------------------------------------------

    def add_cluster(self, cluster_id, **labels):
        """Add a new cluster in the cluster and similarity views."""

        if cluster_id in self.cluster_ids:
            logger.warning(f"Cluster {cluster_id} already exists.")
            return

        # Update the data structures.
        self.cluster_ids.append(cluster_id)
        self.cluster_groups[cluster_id] = labels.get('group', None)
        for name, d in self.cluster_labels.items():
            d[cluster_id] = labels.get(name, None)

        # Add the cluster to the tables.
        d = [{'id': cluster_id, **labels}]
        self.cluster_view.add(d)
        self.reset_similarity_view(self.selected_clusters)

    def change_cluster(self, cluster_id, **labels):
        """Change an existing cluster in the cluster and similarity views."""
        d = [{'id': cluster_id, **labels}]
        self.cluster_view.change(d)
        self.similarity_view.change(d)

    def remove_cluster(self, cluster_id):
        """Remove a cluster from the cluster and similarity views."""
        self.cluster_view.remove([cluster_id])
        self.similarity_view.remove([cluster_id])

    def add_column(self, col_name):
        """Add a column."""
        self.cluster_view.add_column(col_name)
        self.similarity_view.add_column(col_name)

    def remove_column(self, col_name):
        """Remove a column."""
        self.cluster_view.remove_column(col_name)
        self.similarity_view.remove_column(col_name)


# -----------------------------------------------------------------------------
# Supervisor
# -----------------------------------------------------------------------------

class Supervisor:
    pass
