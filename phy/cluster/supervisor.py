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
from phy.gui.qt import _block, set_busy, _wait
from phy.gui.widgets import Table, _uniq

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


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

    Events
    ------

    * request_similar_clusters(cluster_id)

    """

    _required_columns = ('n_spikes', 'similarity')
    _view_name = 'similarity_view'
    _selected_index_offset = 0

    def set_selected_index_offset(self, n):
        """Set the index of the selected cluster, used for correct coloring in the similarity
        view."""
        self._selected_index_offset = n

    def reset(self, cluster_ids=None):
        """Recreate the similarity view, given the selected clusters in the cluster view."""
        cluster_ids = cluster_ids or []
        if not len(cluster_ids):
            self.setRowCount(0)
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
# Supervisor
# -----------------------------------------------------------------------------

class Supervisor:
    pass
