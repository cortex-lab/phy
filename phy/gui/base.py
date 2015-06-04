# -*- coding: utf-8 -*-

"""Base classes for GUIs."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import Counter
import inspect

from ..ext.six import string_types
from ..utils._misc import _show_shortcuts
from ..utils import debug, EventEmitter
from ..utils.settings import (Settings,
                              _ensure_dir_exists,
                              _phy_user_dir,
                              )
from .static import _wrap_html
from .dock import DockWindow


#------------------------------------------------------------------------------
# BaseViewModel
#------------------------------------------------------------------------------

class BaseViewModel(object):
    """Interface between a view and a model.

    Events
    ------

    show_view
    close_view

    """
    _view_name = ''
    _imported_params = ('position', 'size',)

    def __init__(self, model=None, **kwargs):
        self._model = model
        self._event = EventEmitter()

        # Instantiate the underlying view.
        self._view = self._create_view(**kwargs)

        # Set passed keyword arguments as attributes.
        for param in self.imported_params():
            value = kwargs.get(param, None)
            if value is not None:
                setattr(self, param, value)

        self.on_open()

    def emit(self, *args, **kwargs):
        self._event.emit(*args, **kwargs)

    def connect(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    # Methods to override
    #--------------------------------------------------------------------------

    def _create_view(self, **kwargs):
        """Create the view with the parameters passed to the constructor.

        Must be overriden."""
        return None

    def on_open(self):
        """Initialize the view after the model has been loaded.

        May be overriden."""

    def on_close(self):
        """Called when the model is closed.

        May be overriden."""

    # Parameters
    #--------------------------------------------------------------------------

    @classmethod
    def imported_params(cls):
        """All parameter names to be imported on object creation."""
        out = ()
        for base_class in inspect.getmro(cls):
            if base_class == object:
                continue
            out += base_class._imported_params
        return out

    def exported_params(self, save_size_pos=True):
        """Return a dictionary of variables to save when the view is closed."""
        if save_size_pos and hasattr(self._view, 'pos'):
            return {
                'position': (self._view.x(), self._view.y()),
                'size': (self._view.width(), self._view.height()),
            }
        else:
            return {}

    @classmethod
    def get_params(cls, settings):
        """Return the parameter values for the creation of the view."""
        name = cls._view_name
        param_names = cls.imported_params()
        params = {key: settings[name + '_' + key]
                  for key in param_names
                  if (name + '_' + key) in settings}
        return params

    # Properties
    #--------------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._view_name

    @property
    def view(self):
        return self._view

    # Public methods
    #--------------------------------------------------------------------------

    def close(self):
        self._view.close()
        self.emit('close_view')

    def show(self):
        """Show the view."""
        self._view.show()
        self.emit('show_view')


#------------------------------------------------------------------------------
# HTMLViewModel
#------------------------------------------------------------------------------

class HTMLViewModel(BaseViewModel):
    """Widget with custom HTML code.

    To create a new HTML view, derive from `HTMLViewModel`, and implement
    `get_html()` which returns HTML code.

    """
    _html = ''

    def _update(self, view, **kwargs):
        view.setHtml(_wrap_html(html=self.get_html(**kwargs)))

    def _create_view(self, **kwargs):
        from PyQt4.QtWebKit import QWebView
        if 'html' in kwargs:
            self._html = kwargs['html']
        view = QWebView()
        if self._html:
            self._update(view, **kwargs)
        return view

    def get_html(self, **kwargs):
        """Return the HTML contents of the view.

        Must be overriden.

        """
        return self._html

    def update(self, **kwargs):
        """Update the widget's HTML contents."""
        self._update(self._view, **kwargs)

    def isVisible(self):
        return self._view.isVisible()


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

    def _widget_name(self, widget):
        if widget.name:
            return widget.name
        # Fallback to the name given in widget_classes.
        for name, cls in self._widget_classes.items():
            if cls == widget.__class__:
                return name

    def get(self, name=None):
        """Return the list of widgets of a given type."""
        if name is None:
            return self._widgets
        return [widget for widget in self._widgets
                if self._widget_name(widget) == name]

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
        def on_close(e=None):
            self.emit('close', widget)
            self.remove(widget)

        if show:
            widget.show()

        return widget

    def remove(self, widget):
        if widget in self._widgets:
            debug("Remove widget {}.".format(widget))
            self._widgets.remove(widget)
        else:
            debug("Unable to remove widget {}.".format(widget))


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
    state : object
        Default Qt GUI state.
    shortcuts : dict
        Dictionary `{function_name: keyboard_shortcut}`.

    Events
    ------

    add_view
    close_view
    reset_gui
    close_gui

    """

    def __init__(self,
                 model=None,
                 vm_classes=None,
                 state=None,
                 shortcuts=None,
                 config=None,
                 ):
        super(BaseGUI, self).__init__()
        if state is None:
            state = {}
        self.model = model
        self._shortcuts = shortcuts or {}
        self._config = config or [(vm.name, {}) for vm in (vm_classes or {})]
        self._dock = DockWindow(title=self.title)
        self._view_creator = WidgetCreator(widget_classes=vm_classes)
        self._load_config(config,
                          requested_count=state.get('view_count', None),
                          )
        self._load_geometry_state(state)
        # Default close shortcut.
        if 'close' not in self._shortcuts:
            self._shortcuts['close'] = 'ctrl+q'
            self._add_gui_shortcut('close')
        self._create_actions()
        self._set_default_view_connections()
        self.on_open()

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

    def _create_actions(self):
        """Create default actions in the GUI.

        The `_add_gui_shortcut()` method can be used.

        Must be overriden.

        """
        pass

    def on_open(self):
        """Callback function when the model is opened.

        Must be overriden.

        """
        pass

    #--------------------------------------------------------------------------
    # Internal methods
    #--------------------------------------------------------------------------

    def _load_config(self, config=None,
                     current_count=None,
                     requested_count=None):
        """Load a GUI configuration dictionary."""
        config = config or []
        current_count = current_count or {}
        requested_count = requested_count or Counter([name
                                                      for name, _ in config])
        for name, kwargs in config:
            # Add the right number of views of each type.
            if current_count.get(name, 0) >= requested_count.get(name, 0):
                continue
            debug("Adding {} view in GUI.".format(name))
            # GUI-specific keyword arguments position, size, maximized
            position = kwargs.pop('position', None)
            vm = self._view_creator.add(name, **kwargs)
            self.add_view(vm, title=name.capitalize(), position=position)
            if name not in current_count:
                current_count[name] = 0
            current_count[name] += 1
        assert current_count == requested_count

    def _load_geometry_state(self, gui_state):
        if gui_state:
            self._dock.restore_geometry_state(gui_state)

    def _add_gui_shortcut(self, method_name):
        """Helper function to add a GUI action with a keyboard shortcut."""
        # Get the keyboard shortcut for this method.
        shortcut = self._shortcuts.get(method_name, None)
        # Bind the shortcut to the method.
        self._dock.add_action(method_name,
                              getattr(self, method_name),
                              shortcut=shortcut,
                              )

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
        """Add a new view instance to the GUI."""
        position = kwargs.pop('position', None)
        # Item may be a string.
        if isinstance(item, string_types):
            name = item
            item = self._view_creator.add(item, **kwargs)
            # Set the view name if necessary.
            if not item._view_name:
                item._view_name = name
        # Default dock title.
        if title is None:
            title = _title(item)
        # Get the underlying view.
        view = item.view if isinstance(item, BaseViewModel) else item
        # Add the view to the main window.
        dw = self._dock.add_view(view,
                                 title=title,
                                 position=position,
                                 )

        # Dock widget close event.
        @dw.connect_
        def on_close_widget():
            # debug("Close dock widget {}.".format(item))
            self._view_creator.remove(item)
            self.emit('close_view', item)

        # Make sure the callback above is called when the dock widget
        # is closed directly.
        # View model close event.
        @item.connect
        def on_close_view(e=None):
            dw.close()

        self.emit('add_view', view)

    def get_views(self, name=None):
        """Return the list of views of a given type."""
        return self._view_creator.get(name=name)

    @property
    def views(self):
        return self.get_views()

    def view_count(self):
        return {name: len(self.get_views(name))
                for name in self._view_creator.widget_classes.keys()}

    def reset_gui(self):
        """Reset the GUI configuration."""
        count = self.view_count()
        self._load_config(self._config,
                          current_count=count,
                          )
        self.emit('reset_gui')

    def show_shortcuts(self):
        """Show the list of all keyboard shortcuts."""
        _show_shortcuts(self._shortcuts, name=self.__class__.__name__)

    def isVisible(self):
        return self._dock.isVisible()

    def close(self):
        """Close the GUI."""
        self.emit('close_gui')
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

        self._create_settings(default_settings_path)

        self._view_creator = WidgetCreator(widget_classes=vm_classes)

        if gui_classes is None:
            gui_classes = self.settings['gui_classes']
        self._gui_creator = WidgetCreator(widget_classes=gui_classes)

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

    def _save_model(self):
        """Save a model.

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

    def save(self):
        self._save_model()

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
        params = {p: self.settings.get('{}_{}'.format(name, p), None)
                  for p in ('config', 'shortcuts', 'state')}
        params.update(kwargs)

        # Create the GUI.
        gui = self._gui_creator.add(name, model=self.model, **params)
        gui._save_state = True

        # Connect the 'open' event.
        self.connect(gui.on_open)

        @gui.connect
        def on_close_gui():
            self.unconnect(gui.on_open)
            # Save the params of every view in the GUI.
            for vm in gui.views:
                self.save_view_params(vm, save_size_pos=False)
            gs = gui.main_window.save_geometry_state()
            gs['view_count'] = gui.view_count()
            if not gui._save_state:
                gs['state'] = None
                gs['geometry'] = None
            self.settings['{}_state'.format(name)] = gs
            self.settings.save()

        # HACK: do not save GUI state when views have been closed or reset
        # in the session, otherwise Qt messes things up in the GUI.
        @gui.connect
        def on_close_view(view):
            gui._save_state = False

        @gui.connect
        def on_reset_gui():
            gui._save_state = False

        return gui

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
        if not vm_class._view_name:
            vm_class._view_name = name
        # Get default and user parameters.
        params = vm_class.get_params(self.settings)
        params.update(kwargs)

        vm = self._view_creator.add(vm_class,
                                    model=self.model,
                                    **params)
        # Connect the 'open' event.
        self.connect(vm.on_open)

        # Save the view parameters when the view is closed.
        @vm.connect
        def on_close(e=None):
            self.unconnect(vm.on_open)
            self.save_view_params(vm)

        return vm

    def save_view_params(self, vm, save_size_pos=True):
        """Save the parameters exported by a view model instance."""
        to_save = vm.exported_params(save_size_pos=save_size_pos)
        for key, value in to_save.items():
            assert vm.name
            name = '{}_{}'.format(vm.name, key)
            self.settings[name] = value
            debug("Save {0}={1} for {2}.".format(name, value, vm.name))
