# -*- coding: utf-8 -*-
from __future__ import print_function

"""GUI creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ...utils.dock import DockWindow
from ...utils import EventEmitter, debug
from ...plot.view_models import BaseViewModel


#------------------------------------------------------------------------------
# Manual clustering window
#------------------------------------------------------------------------------

class KlustaViewa(EventEmitter):
    def __init__(self, session, config=None):
        self.session = session
        self._dock = DockWindow()
        self._load_config(config)
        # Save the geometry state
        self._dock.on_close(self.on_close)
        self._create_gui_actions()
        self._set_default_view_connections()
        self._cluster_ids = []
        self.start()

    def _load_config(self, config=None):
        if config is None:
            config = self.session['gui_config']
        for name, kwargs in config:
            # GUI-specific keyword arguments position, size, maximized
            position = kwargs.pop('position', None)
            item = self.session.view_creator.add(name, **kwargs)
            self.add_view(item, position=position)

        # Load geometry state
        gs = self.session.settings['gui_state']
        if gs:
            self._dock.restore_geometry_state(gs)

    def start(self):
        self.session.wizard.start()
        self._cluster_ids = [self.session.wizard.best]

    def on_close(self):
        gs = self._dock.save_geometry_state()
        self.session.settings['gui_state'] = gs

    @property
    def main_window(self):
        return self._dock

    def add_view(self, item):
        if isinstance(item, BaseViewModel):
            view = item.view
            dw = self._dock.add_view(view)

            self.connect(item.on_select)

            # Make sure the dock widget is closed when the view it contains
            # is closed with the Escape key.
            @view.connect
            def on_close(e):
                self.unconnect(item.on_select)
                dw.close()

        else:
            self._dock.add_view(item)

    def get_views(self, name):
        vms = self.session.gui_creator.get(name)
        dock_widgets = [_.widget() for _ in self._dock.list_views()]
        return [vm for vm in vms if vm.view.native in dock_widgets]

    def _set_default_view_connections(self):

        # Select feature dimension from waveform view.
        @self.connect_views('waveforms', 'features')
        def channel_click(waveforms, features):

            @waveforms.connect
            def on_channel_click(e):
                if e.key in map(str, range(10)):
                    channel = e.channel_idx
                    dimension = int(e.key.name)
                    feature = 0 if e.button == 1 else 1
                    if (0 <= dimension <= len(features.dimensions) - 1):
                        features.dimensions[dimension] = (channel, feature)
                        # Force view update.
                        features.dimensions = features.dimensions

    def _create_gui_actions(self):

        shortcuts = self.session.settings['keyboard_shortcuts']

        def _add_gui_shortcut(func):
            """Helper function to add a GUI action with a keyboard shortcut."""
            name = func.__name__
            shortcut = shortcuts.get(name, None)
            self._dock.shortcut(name, shortcut)(func)

        # Update the wizard selection after a clustering action.
        @self.session.connect
        def on_cluster(up):
            self._wizard_select()

        # Move best/match/both to noise/mua/good
        # ---------------------------------------------------------------------

        def _get_clusters(which):
            return {
                'best': [self.session.wizard.best],
                'match': [self.session.wizard.match],
                'both': [self.session.wizard.best, self.session.wizard.match],
            }[which]

        def _make_func(which, group):
            """Return a function that moves best/match/both clusters to
            a group."""

            def func():
                clusters = _get_clusters(which)
                if None in clusters:
                    return
                self.session.move(clusters, group)

            func.__name__ = 'move_{}_to_{}'.format(which, group)
            return func

        for which in ('best', 'match', 'both'):
            for group in ('noise', 'mua', 'good'):
                _add_gui_shortcut(_make_func(which, group))

    # General actions
    # ---------------------------------------------------------------------

    def reset_gui(self):
        # TODO
        pass
        # # Add missing views.
        # present = set(self._dock.view_counts())
        # default = set(self._default_counts)
        # to_add = default - present
        # counts = {name: 1 for name in to_add}
        # # Add the default views.
        # self._add_gui_views(gui, self._cluster_ids, counts=counts)

    def save(self):
        self.session.save()

    def undo(self):
        self.session.undo()

    def redo(self):
        self.session.redo()

    def show_shortcuts(self):
        shortcuts = self.session.settings['keyboard_shortcuts']
        for name in sorted(shortcuts):
            print("{0:<24}: {1:s}".format(name, str(shortcuts[name])))

    def exit(self):
        self._dock.close()

    # Selection
    # ---------------------------------------------------------------------

    def select(self, cluster_ids):
        cluster_ids = list(cluster_ids)
        assert len(cluster_ids) == len(set(cluster_ids))
        # Do not re-select an already-selected list of clusters.
        if cluster_ids == self._cluster_ids:
            return
        assert set(cluster_ids) <= set(self.session.clustering.cluster_ids)
        debug("Select clusters {0:s}.".format(str(cluster_ids)))
        self._cluster_ids = cluster_ids
        self.emit('select', cluster_ids)

    # Wizard list
    # ---------------------------------------------------------------------

    def _wizard_select(self):
        self.select(self.session.wizard.selection)

    def reset_wizard(self):
        self.session.wizard.start()
        self._wizard_select()

    def first(self):
        self.session.wizard.first()
        self._wizard_select()

    def last(self):
        self.session.wizard.last()
        self._wizard_select()

    def next(self):
        self.session.wizard.next()
        self._wizard_select()

    def previous(self):
        self.session.wizard.previous()
        self._wizard_select()

    def pin(self):
        self.session.wizard.pin()
        self._wizard_select()

    def unpin(self):
        self.session.wizard.unpin()
        self._wizard_select()

    # Cluster actions
    # ---------------------------------------------------------------------

    def merge(self):
        clusters = self._cluster_ids
        if len(clusters) >= 2:
            self.merge(clusters)

    def split(self):
        # TODO: refactor
        pass


#------------------------------------------------------------------------------
# GUI creator
#------------------------------------------------------------------------------

class GUICreator(object):
    def __init__(self, session):
        self.session = session
        self._guis = []

    @property
    def default_config(self):
        # TODO: find default config in user params
        return []

    def add(self, config=None, show=True):
        gui = KlustaViewa(self.session, config=config)
        self._guis.append(gui)

        @gui.on_close
        def on_close():
            self._guis.remove(gui)

        return gui

    @property
    def guis(self):
        return self._guis

    @property
    def gui(self):
        if len(self._guis) != 1:
            return
        return self._guis[0]
