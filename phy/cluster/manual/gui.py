# -*- coding: utf-8 -*-
from __future__ import print_function

"""GUI creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import phy
from ...utils._misc import _show_shortcuts
from ...utils.dock import DockWindow, _create_web_view, _prompt
from ...utils import EventEmitter, debug
from ...plot.view_models import BaseViewModel


#------------------------------------------------------------------------------
# Manual clustering window
#------------------------------------------------------------------------------

class ClusterManualGUI(EventEmitter):
    """Manual clustering GUI.

    This object represents a main window with:

    * multiple views
    * a wizard panel
    * high-level clustering methods
    * global keyboard shortcuts

    """

    def __init__(self, session, config=None):
        super(ClusterManualGUI, self).__init__()
        self.session = session
        self.start()
        self._dock = DockWindow(title=self.title)
        # Load the saved view count.
        vc = self.session.settings.get('gui_view_count', None)
        # Default GUI config.
        config = config or self.session.settings['gui_config']
        # Remove non-existing views.
        if vc and config:
            config = [(name, _) for (name, _) in config
                      if name in vc]
        # Create the views.
        self._load_config(config)
        self._load_geometry_state()
        self._create_gui_actions()
        self._set_default_view_connections()

    def _load_config(self, config=None):
        """Load a GUI configuration dictionary."""
        for name, kwargs in config:
            debug("Adding {} view in GUI.".format(name))
            # GUI-specific keyword arguments position, size, maximized
            position = kwargs.pop('position', None)
            if name == 'wizard':
                item = self._create_wizard_panel()
            else:
                clusters = self._cluster_ids
                item = self.session.view_creator.add(name,
                                                     cluster_ids=clusters,
                                                     **kwargs)
            self.add_view(item, title=name.capitalize(), position=position)

    def _load_geometry_state(self):
        # Load geometry state
        gs = self.session.settings.get('gui_state', None)
        if gs:
            self._dock.restore_geometry_state(gs)

    def show(self):
        """Show the GUI"""
        self._dock.show()

    def start(self):
        """Start the wizard."""
        self.session.wizard.start()
        self._cluster_ids = self.session.wizard.selection

    @property
    def main_window(self):
        """Dock main window."""
        return self._dock

    @property
    def title(self):
        """Title of the main window."""
        name = self.__class__.__name__
        filename = self.session.model.kwik_path
        clustering = self.session.model.clustering
        channel_group = self.session.model.channel_group
        template = ("{filename} (shank {channel_group}, "
                    "{clustering} clustering) "
                    "- {name} - phy {version}")
        return template.format(name=name,
                               version=phy.__version__,
                               filename=filename,
                               channel_group=channel_group,
                               clustering=clustering,
                               )

    def add_view(self, item, title=None, **kwargs):
        """Add a new view model instance to the GUI."""
        view = item.view if isinstance(item, BaseViewModel) else item
        # Default dock title.
        if title is None:
            if hasattr(item, 'name'):
                title = item.name.capitalize()
            else:
                title = item.__class__.__name__.capitalize()
        dw = self._dock.add_view(view, title=title, **kwargs)

        if not isinstance(item, BaseViewModel):
            return

        @self.connect
        def on_select(cluster_ids):
            item.select(cluster_ids)

        # Make sure the dock widget is closed when the view it contains
        # is closed with the Escape key.
        @view.connect
        def on_close(e):
            self.unconnect(item.on_select)
            dw.close()

    @property
    def dock_widgets(self):
        return [_.widget() for _ in self._dock.list_views()]

    def get_views(self, name=None):
        """Return the list of views of a given type."""
        vms = self.session.view_creator.get(name)
        # Among all created view models, return those that are in the GUI.
        return [vm for vm in vms if vm.view.native in self.dock_widgets]

    def _set_default_view_connections(self):
        """Set view connections."""

        # Select feature dimension from waveform view.
        @self._dock.connect_views('waveforms', 'features')
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

        for name in ['reset_gui',
                     'save',
                     'undo',
                     'redo',
                     'show_shortcuts',
                     'exit',
                     'select',
                     'reset_wizard',
                     'first',
                     'last',
                     'next',
                     'previous',
                     'pin',
                     'unpin',
                     'merge',
                     'split',
                     ]:
            _add_gui_shortcut(getattr(self, name))

        # Update the wizard selection after a clustering action.
        @self.session.connect
        def on_cluster(up):
            # Special case: split.
            if not up.history and up.description == 'assign':
                self.select(up.added)
            else:
                self._wizard_select()

        # Move best/match/both to noise/mua/good.
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

    def _create_wizard_panel(self, cluster_ids=None):
        styles = '''

        html, body, div {
            background-color: black;
        }

        .control-panel {
            background-color: black;
            color: white;
        }

        '''

        def _get_html():
            return self.session.wizard.get_panel(extra_styles=styles)

        view = _create_web_view(_get_html())

        @self.connect
        def on_select(cluster_ids):
            view.setHtml(_get_html())

        @self.connect
        def on_cluster(up):
            view.setHtml(_get_html())

        return view

    # General actions
    # ---------------------------------------------------------------------

    def reset_gui(self):
        """Reset the GUI configuration."""
        config = self.session.settings['gui_config']
        existing = sorted(self._dock.view_counts())
        to_add = [(name, _) for (name, _) in config if name not in existing]
        self._load_config(to_add)
        self.session.settings['gui_state'] = None

    def save(self):
        """Save the clustering changes to the `.kwik` file."""
        self.session.save()

    def undo(self):
        """Undo the last clustering action."""
        self.session.undo()

    def redo(self):
        """Redo the last clustering action."""
        self.session.redo()

    def show_shortcuts(self):
        """Show the list off all keyboard shortcuts."""
        shortcuts = self.session.settings['keyboard_shortcuts']
        _show_shortcuts(shortcuts, name=self.__class__.__name__)

    def close(self):
        """Close the GUI."""
        if (self.session.settings['prompt_save_on_exit'] and
                self.session.has_unsaved_changes):
            res = _prompt(self._dock,
                          "Do you want to save your changes?",
                          ('save', 'cancel', 'close'))
            if res == 'save':
                self.save()
            elif res == 'cancel':
                return
            elif res == 'close':
                pass
        self._dock.close()

    def exit(self):
        """Close the GUI."""
        self.close()

    # Selection
    # ---------------------------------------------------------------------

    def select(self, cluster_ids):
        """Select clusters."""
        cluster_ids = list(cluster_ids)
        assert len(cluster_ids) == len(set(cluster_ids))
        # Do not re-select an already-selected list of clusters.
        if cluster_ids == self._cluster_ids:
            return
        assert set(cluster_ids) <= set(self.session.clustering.cluster_ids)
        debug("Select clusters {0:s}.".format(str(cluster_ids)))
        self._cluster_ids = cluster_ids
        self.emit('select', cluster_ids)

    @property
    def selected_clusters(self):
        """The list of selected clusters."""
        return self._cluster_ids

    # Wizard list
    # ---------------------------------------------------------------------

    def _wizard_select(self):
        self.select(self.session.wizard.selection)

    def reset_wizard(self):
        """Restart the wizard."""
        self.session.wizard.start()
        self._wizard_select()

    def first(self):
        """Go to the first cluster proposed by the wizard."""
        self.session.wizard.first()
        self._wizard_select()

    def last(self):
        """Go to the last cluster proposed by the wizard."""
        self.session.wizard.last()
        self._wizard_select()

    def next(self):
        """Go to the next cluster proposed by the wizard."""
        self.session.wizard.next()
        self._wizard_select()

    def previous(self):
        """Go to the previous cluster proposed by the wizard."""
        self.session.wizard.previous()
        self._wizard_select()

    def pin(self):
        """Pin the current best cluster."""
        self.session.wizard.pin()
        self._wizard_select()

    def unpin(self):
        """Unpin the current best cluster."""
        self.session.wizard.unpin()
        self._wizard_select()

    # Cluster actions
    # ---------------------------------------------------------------------

    def merge(self):
        """Merge all selected clusters together."""
        clusters = self._cluster_ids
        if len(clusters) >= 2:
            self.session.merge(clusters)

    def split(self):
        """Create a new cluster out of the selected spikes."""
        for features in self.get_views('features'):
            spikes = features.spikes_in_lasso()
            if spikes is not None:
                self.session.split(spikes)
                features.lasso.clear()
                return


#------------------------------------------------------------------------------
# GUI creator
#------------------------------------------------------------------------------

class GUICreator(object):
    def __init__(self, session):
        self.session = session
        self._guis = []

    def add(self, config=None, show=True):
        """Add a new manual clustering GUI.

        Parameters
        ----------

        config : list
            A list of tuples `(name, kwargs)` describing the views in the GUI.
        show : bool
            Whether to show the newly-created GUI.

        Returns
        -------

        gui : ClusterManualGUI
            The GUI.

        """
        gui = ClusterManualGUI(self.session, config=config)
        self._guis.append(gui)

        @gui.main_window.on_close
        def on_close():
            if gui in self._guis:
                self._guis.remove(gui)
            self.session.view_creator.save_view_params()
            gs = gui._dock.save_geometry_state()
            self.session.settings['gui_state'] = gs
            self.session.settings['gui_view_count'] = gui._dock.view_counts()
            self.session.settings.save()

        if show:
            gui.show()

        return gui

    @property
    def guis(self):
        """List of GUIs."""
        return self._guis

    @property
    def gui(self):
        """The GUI if there is only one."""
        if len(self._guis) != 1:
            return
        return self._guis[0]
