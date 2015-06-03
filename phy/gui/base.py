# -*- coding: utf-8 -*-

"""Base classes for GUIs."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..ext.six import string_types
from ..utils._misc import _show_shortcuts
from ..utils import debug, EventEmitter
from ..utils.settings import (Settings,
                              _ensure_dir_exists,
                              _phy_user_dir,
                              )
from .dock import DockWindow
from ..plot.view_models.base import BaseViewModel


#------------------------------------------------------------------------------
# Widget creator (base class for ViewCreator and GUICreator)
#------------------------------------------------------------------------------

class WidgetCreator(object):
    """Manage the creation of widgets.

    A widget must implement:

    * `name`
    * `show()`
    * `connect` (for `close` event)

    """
    def __init__(self, widget_classes=None):
        self._widget_classes = widget_classes or {}
        self._widgets = []

    def _create_widget(self, widget_class, **kwargs):
        """Create a new widget of a given class.

        Must be overriden.

        """
        pass

    @property
    def widget_classes(self):
        return self._widget_classes

    def get(self, name=None):
        """Return the list of widgets of a given type."""
        if name is None:
            return self._widgets
        return [widget for widget in self._widgets if widget.name == name]

    def add(self, widget_class, show=True, **kwargs):
        """Add a new widget."""
        # widget_class can also be a name, but in this case it must be
        # registered in self._widget_classes.
        if isinstance(widget_class, string_types):
            widget_class = self.widget_classes.get(widget_class)
        widget = self._create_widget(widget_class, **kwargs)

        if widget not in self._widgets:
            self._widgets.append(widget)

        @widget.connect
        def on_close(event):
            self._widgets.remove(widget)

        if show:
            widget.show()

        return widget


#------------------------------------------------------------------------------
# View creator
#------------------------------------------------------------------------------

class ViewCreator(WidgetCreator):
    """Create views from a model."""

    def __init__(self, session, vm_classes=None, save_size_pos=True):
        super(ViewCreator, self).__init__(widget_classes=vm_classes)
        self.session = session
        self._save_size_pos = save_size_pos

    def _create_widget(self, vm_class, **kwargs):
        """Create a new view model instance."""

        # Load parameters from the settings.
        params = vm_class.get_params(self.session.settings)
        params.update(kwargs)

        vm = vm_class(model=self.session.model,
                      store=self.session.cluster_store,
                      **params)

        self.session.connect(vm.on_open)

        @vm.connect
        def on_close(event):
            self.session.unconnect(vm.on_open)
            self._save_vm_params(vm)

        return vm

    def _save_vm_params(self, vm):
        """Save the parameters exported by a view model instance."""
        to_save = vm.exported_params(self._save_size_pos)
        for key, value in to_save.items():
            name = '{}_{}'.format(vm.name, key)
            self.session.settings[name] = value
            debug("Save {0}={1} for {2}.".format(name, value, vm.name))

    def save_view_params(self):
        """Save all view parameters to user settings."""
        for vm in self._vms:
            self._save_vm_params(vm)


#------------------------------------------------------------------------------
# GUI creator
#------------------------------------------------------------------------------

class GUICreator(WidgetCreator):
    def __init__(self, session, gui_classes=None):
        super(GUICreator, self).__init__(widget_classes=gui_classes)
        self.session = session

    def _create_widget(self, widget_class, **kwargs):
        gui = widget_class(self.session, **kwargs)

        @gui.connect
        def on_close():
            gui.view_creator.save_view_params()
            gs = gui.main_window.save_geometry_state()
            vc = gui.main_window.view_counts()
            self.session.settings['gui_state'] = gs
            self.session.settings['gui_view_count'] = vc
            self.session.settings.save()

        return gui

    @property
    def guis(self):
        """List of GUIs."""
        return self._widgets

    @property
    def gui(self):
        """The GUI if there is only one."""
        if len(self.guis) != 1:
            return
        return self.guis[0]


#------------------------------------------------------------------------------
# Base GUI
#------------------------------------------------------------------------------

def _title(item):
    """Default view model title."""
    if hasattr(item, 'name'):
        return item.name.capitalize()
    else:
        return item.__class__.__name__.capitalize()


class BaseGUI(EventEmitter):
    """Base GUI.

    This object represents a main window with:

    * multiple dockable views
    * user-exposed actions
    * keyboard shortcuts

    Parameters
    ----------

    config : list
        List of pairs `(name, kwargs)` to create default views.
    vm_classes : dict
        Dictionary `{name: view_model_class}`.
    gui_state : object
        Default Qt GUI state.
    shortcuts : dict
        Dictionary `{function_name: keyboard_shortcut}`.

    Events
    ------

    add_view(view)
    reset_gui()

    """

    def __init__(self,
                 config=None,
                 vm_classes=None,
                 gui_state=None,
                 shortcuts=None,
                 ):
        super(BaseGUI, self).__init__()
        self._shortcuts = {}
        self._config = config
        self._dock = DockWindow(title=self.title)
        self._view_creator = ViewCreator(vm_classes=vm_classes)
        self._load_config(config)
        self._load_geometry_state(gui_state)
        self._create_gui_actions()
        self._set_default_view_connections()

    #--------------------------------------------------------------------------
    # Methods to override
    #--------------------------------------------------------------------------

    @property
    def title(self):
        """Title of the main window.

        May be overriden.

        """
        return 'Base GUI'

    def _set_default_view_connections(self):
        """Set view connections.

        May be overriden.

        Example:

        ```python
        @self.main_window.connect_views('view_1', 'view_2')
        def f(view_1, view_2):
            # Called for every pair of views of type view_1 and view_2.
            pass
        ```

        """
        pass

    #--------------------------------------------------------------------------
    # Internal methods
    #--------------------------------------------------------------------------

    def _load_config(self, config=None):
        """Load a GUI configuration dictionary."""
        for name, kwargs in config:
            debug("Adding {} view in GUI.".format(name))
            # GUI-specific keyword arguments position, size, maximized
            position = kwargs.pop('position', None)
            vm = self._view_creator.add(name, **kwargs)
            self.add_view(vm, title=name.capitalize(), position=position)

    def _load_geometry_state(self, gui_state):
        if gui_state:
            self._dock.restore_geometry_state(gui_state)

    def _add_gui_shortcut(self, method_name):
        """Helper function to add a GUI action with a keyboard shortcut."""
        # Get the keyboard shortcut for this method.
        shortcut = self._shortcuts.get(method_name, None)
        # Bind the shortcut to the method.
        self._dock.shortcut(method_name, shortcut)(getattr(self, method_name))

    #--------------------------------------------------------------------------
    # Public methods
    #--------------------------------------------------------------------------

    def show(self):
        """Show the GUI"""
        self._dock.show()

    @property
    def main_window(self):
        """Main Qt window."""
        return self._dock

    def add_view(self, item, title=None, **kwargs):
        """Add a new view model instance to the GUI."""
        position = kwargs.pop('position', None)
        # Item may be a string.
        if isinstance(item, string_types):
            item = self._view_creator.add(item, **kwargs)
        # Default dock title.
        if title is None:
            title = _title(item)
        # Get the underlying view.
        view = item.view if isinstance(item, BaseViewModel) else item
        # Add the view to the main window.
        self._dock.add_view(view, title=title, position=position)
        self.emit('add_view', view)

    def get_views(self, name=None):
        """Return the list of views of a given type."""
        return self._view_creator.get(name=name)

    def reset_gui(self):
        """Reset the GUI configuration."""
        existing = sorted(self._dock.view_counts())
        to_add = [(name, _) for (name, _) in self._config
                  if name not in existing]
        self._load_config(to_add)
        self.emit('reset_gui')

    def show_shortcuts(self):
        """Show the list of all keyboard shortcuts."""
        _show_shortcuts(self._shortcuts, name=self.__class__.__name__)

    def close(self):
        """Close the GUI."""
        self.emit('close')
        self._dock.close()

    def exit(self):
        """Close the GUI."""
        self.close()


#------------------------------------------------------------------------------
# Session
#------------------------------------------------------------------------------

class BaseSession(EventEmitter):
    """Give access to the data, views, and GUIs in an interactive session.

    The model must implement:

    * `model(path)`
    * `model.path`
    * `model.close()`

    """
    def __init__(self,
                 model=None,
                 path=None,
                 phy_user_dir=None,
                 default_settings_path=None,
                 vm_classes=None,
                 gui_classes=None,
                 ):
        super(BaseSession, self).__init__()
        self.model = None
        if phy_user_dir is None:
            phy_user_dir = _phy_user_dir()
        _ensure_dir_exists(phy_user_dir)
        self.phy_user_dir = phy_user_dir
        self.view_creator = ViewCreator(self, vm_classes=vm_classes)
        self.gui_creator = GUICreator(self, gui_classes=gui_classes)
        self._create_settings(default_settings_path)

        self.connect(self.on_open)
        self.connect(self.on_close)

        if model:
            self.open(path=path, model=model)

    def _create_settings(self, default_settings_path):
        self.settings = Settings(phy_user_dir=self.phy_user_dir,
                                 default_path=default_settings_path,
                                 )

        @self.connect
        def on_open():
            # Initialize the settings with the model's path.
            self.settings.on_open(self.model.path)

    # File-related actions
    # -------------------------------------------------------------------------

    def _create_model(self, path):
        """Create a model from a path.

        Must be overriden.

        """
        pass

    def open(self, path=None, model=None):
        """Open a dataset."""
        # Close the session if it is already open.
        if self.model:
            self.close()
        if model is None:
            model = self._create_model(path)
        self.model = model
        self.emit('open')

    def close(self):
        """Close the currently-open dataset."""
        self.model.close()
        self.emit('close')
        self.model = None
        self.experiment_path = None

    # Views and GUIs
    # -------------------------------------------------------------------------

    def show_gui(self, config=None, **kwargs):
        """Show a new manual clustering GUI."""
        # Ensure that a Qt application is running.
        gui = self.gui_creator.add(config, **kwargs)
        return gui

    def show_view(self, name, cluster_ids, **kwargs):
        """Create and display a new view.

        Parameters
        ----------

        name : str
            Can be `waveforms`, `features`, `correlograms`, or `traces`.
        cluster_ids : array-like
            List of clusters to show.

        Returns
        -------

        vm : `ViewModel` instance

        """
        show = kwargs.pop('show', True)
        vm = self.view_creator.add(name,
                                   show=show,
                                   cluster_ids=cluster_ids,
                                   **kwargs)
        return vm
