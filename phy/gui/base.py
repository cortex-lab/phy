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
# Widget creator (used to create views and GUIs)
#------------------------------------------------------------------------------

class WidgetCreator(EventEmitter):
    """Manage the creation of widgets.

    A widget must implement:

    * `name`
    * `show()`
    * `connect` (for `close` event)

    Events
    ------

    add(widget): when a widget is added.
    close(widget): when a widget is closed.

    """
    def __init__(self, widget_classes=None):
        super(WidgetCreator, self).__init__()
        self._widget_classes = widget_classes or {}
        self._widgets = []

    def _create_widget(self, widget_class, **kwargs):
        """Create a new widget of a given class.

        May be overriden.

        """
        return widget_class(**kwargs)

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
            if widget_class not in self.widget_classes:
                raise ValueError("Unknown widget class "
                                 "`{}`.".format(widget_class))
            widget_class = self.widget_classes[widget_class]
        widget = self._create_widget(widget_class, **kwargs)
        if widget not in self._widgets:
            self._widgets.append(widget)
        self.emit('add', widget)

        @widget.connect
        def on_close(event):
            self.emit('close', widget)
            self._widgets.remove(widget)

        if show:
            widget.show()

        return widget


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
    close()

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
        self._view_creator = WidgetCreator(vm_classes=vm_classes)
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

    @property
    def views(self):
        return self.get_views()

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
        self.view_creator = WidgetCreator(widget_classes=vm_classes)
        self.gui_creator = WidgetCreator(widget_classes=gui_classes)
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

    def show_gui(self, name, **kwargs):
        """Show a new GUI."""
        # Get the default GUI config.
        params = {'config': self.settings['{}_{}'.format(name, 'config')]}
        params.update(kwargs)
        # Create the GUI.
        gui = self._gui_creator.add(name, **params)
        # Connect the 'open' event.
        self.connect(gui.on_open)

        @gui.connect
        def on_close():
            self.unconnect(gui.on_open)
            # Save the params of every view in the GUI.
            for vm in gui.views:
                self.save_view_params(vm)
            gs = gui.main_window.save_geometry_state()
            vc = gui.main_window.view_counts()
            self.settings['gui_state'] = gs
            self.settings['gui_view_count'] = vc
            self.settings.save()

    def show_view(self, name, **kwargs):
        """Create and display a new view.

        Parameters
        ----------

        name : str
            A view model name.

        Returns
        -------

        vm : `ViewModel` instance

        """
        # Get the view class.
        vm_class = self._view_creator.widget_classes[name]
        # Get default and user parameters.
        params = vm_class.get_params(self.session.settings)
        params.update(kwargs)

        vm = self._view_creator.add(vm_class,
                                    model=self._model,
                                    store=self._store,
                                    **params)
        # Connect the 'open' event.
        self.connect(vm.on_open)

        # Save the view parameters when the view is closed.
        @vm.connect
        def on_close(event):
            self.unconnect(vm.on_open)
            self.save_view_params(vm)

        return vm

    def save_view_params(self, vm):
        """Save the parameters exported by a view model instance."""
        to_save = vm.exported_params(save_size_pos=True)
        for key, value in to_save.items():
            name = '{}_{}'.format(vm.name, key)
            self.settings[name] = value
            debug("Save {0}={1} for {2}.".format(name, value, vm.name))
