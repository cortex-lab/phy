# -*- coding: utf-8 -*-

"""Base classes for GUIs."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import Counter
import inspect

from ..ext.six import string_types
from ..utils._misc import _show_shortcuts
from ..utils import debug, info, warn, EventEmitter
from ._utils import _read
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
        """Emit an event."""
        return self._event.emit(*args, **kwargs)

    def connect(self, *args, **kwargs):
        """Connect a callback function."""
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
        """The model."""
        return self._model

    @property
    def name(self):
        """The view model's name."""
        return self._view_name

    @property
    def view(self):
        """The underlying view."""
        return self._view

    # Public methods
    #--------------------------------------------------------------------------

    def close(self):
        """Close the view."""
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

    def _update(self, view, **kwargs):
        html = self.get_html(**kwargs)
        css = self.get_css(**kwargs)
        wrapped = _read('wrap_qt.html')
        html_wrapped = wrapped.replace('%CSS%', css).replace('%HTML%', html)
        view.setHtml(html_wrapped)

    def _create_view(self, **kwargs):
        from PyQt4.QtWebKit import QWebView
        view = QWebView()
        self._update(view, **kwargs)
        return view

    def get_html(self, **kwargs):
        """Return the non-formatted HTML contents of the view."""
        return ''

    def get_css(self, **kwargs):
        """Return the view's CSS styles."""
        return ''

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
        """The registered widget classes that can be created."""
        return self._widget_classes

    def _widget_name(self, widget):
        if widget.name:
            return widget.name
        # Fallback to the name given in widget_classes.
        for name, cls in self._widget_classes.items():
            if cls == widget.__class__:
                return name

    def get(self, *names):
        """Return the list of widgets of a given type."""
        if not names:
            return self._widgets
        return [widget for widget in self._widgets
                if self._widget_name(widget) in names]

    def add(self, widget_class, show=False, **kwargs):
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
        """Remove a widget."""
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
        name = item.name.capitalize()
    else:
        name = item.__class__.__name__.capitalize()
    return name


def _assert_counters_equal(c_0, c_1):
    c_0 = {(k, v) for (k, v) in c_0.items() if v > 0}
    c_1 = {(k, v) for (k, v) in c_1.items() if v > 0}
    assert c_0 == c_1


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

    """

    _default_shortcuts = {
        'exit': 'ctrl+q',
        'enable_snippet_mode': ':',
    }

    def __init__(self,
                 model=None,
                 vm_classes=None,
                 state=None,
                 shortcuts=None,
                 snippets=None,
                 config=None,
                 settings=None,
                 ):
        super(BaseGUI, self).__init__()
        self.settings = settings or {}
        if state is None:
            state = {}
        self.model = model
        # Shortcuts.
        s = self._default_shortcuts.copy()
        s.update(shortcuts or {})
        self._shortcuts = s
        self._snippets = snippets or {}
        # GUI state and config.
        self._state = state
        if config is None:
            config = [(name, {}) for name in (vm_classes or {})]
        self._config = config
        # Create the dock window.
        self._dock = DockWindow(title=self.title)
        self._view_creator = WidgetCreator(widget_classes=vm_classes)
        self._initialize_views()
        self._load_geometry_state(state)
        self._set_default_shortcuts()
        self._create_actions()
        self._set_default_view_connections()

    def _initialize_views(self):
        self._load_config(self._config,
                          requested_count=self._state.get('view_count', None),
                          )

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

    def _view_model_kwargs(self, name):
        return {}

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
            self.add_view(name, **kwargs)
            if name not in current_count:
                current_count[name] = 0
            current_count[name] += 1
        _assert_counters_equal(current_count, requested_count)

    def _load_geometry_state(self, gui_state):
        if gui_state:
            self._dock.restore_geometry_state(gui_state)

    def _remove_actions(self):
        self._dock.remove_actions()

    def _set_default_shortcuts(self):
        for name, shortcut in self._default_shortcuts.items():
            self._add_gui_shortcut(name)

    def _add_gui_shortcut(self, method_name):
        """Helper function to add a GUI action with a keyboard shortcut."""
        # Get the keyboard shortcut for this method.
        shortcut = self._shortcuts.get(method_name, None)

        def callback():
            return getattr(self, method_name)()

        # Bind the shortcut to the method.
        self._dock.add_action(method_name,
                              callback,
                              shortcut=shortcut,
                              )

    #--------------------------------------------------------------------------
    # Snippet methods
    #--------------------------------------------------------------------------

    @property
    def status_message(self):
        """Message in the status bar."""
        return str(self._dock.status_message)

    @status_message.setter
    def status_message(self, value):
        self._dock.status_message = str(value)

    _snippet_message_cursor = '\u200A\u258C'

    @property
    def _snippet_message(self):
        """This is used to write a snippet message in the status bar.

        A cursor is appended at the end.

        """
        return self.status_message[:-len(self._snippet_message_cursor)]

    @_snippet_message.setter
    def _snippet_message(self, value):
        self.status_message = value + self._snippet_message_cursor

    def process_snippet(self, snippet):
        """Processes a snippet.

        May be overriden.

        """
        assert snippet[0] == ':'
        snippet = snippet[1:]
        split = snippet.split(' ')
        cmd = split[0]
        snippet = snippet[len(cmd):].strip()
        func = self._snippets.get(cmd, None)
        if func is None:
            info("The snippet `{}` could not be found.".format(cmd))
            return
        try:
            info("Processing snippet `{}`.".format(cmd))
            func(self, snippet)
        except Exception as e:
            warn("Error when executing snippet `{}`: {}.".format(
                 cmd, str(e)))

    def _snippet_action_name(self, char):
        return self._snippet_chars.index(char)

    _snippet_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 ._,+*-=:()'

    def _create_snippet_actions(self):
        # One action per allowed character.
        for i, char in enumerate(self._snippet_chars):

            def _make_func(char):
                def callback():
                    self._snippet_message += char
                return callback

            self._dock.add_action('snippet_{}'.format(i),
                                  shortcut=char,
                                  callback=_make_func(char),
                                  )

        def backspace():
            if self._snippet_message == ':':
                return
            self._snippet_message = self._snippet_message[:-1]

        def enter():
            self.process_snippet(self._snippet_message)
            self.disable_snippet_mode()

        self._dock.add_action('snippet_backspace',
                              shortcut='backspace',
                              callback=backspace,
                              )
        self._dock.add_action('snippet_activate',
                              shortcut=('enter', 'return'),
                              callback=enter,
                              )
        self._dock.add_action('snippet_disable',
                              shortcut='escape',
                              callback=self.disable_snippet_mode,
                              )

    def enable_snippet_mode(self):
        info("Snippet mode enabled, press `escape` to leave this mode.")
        self._remove_actions()
        self._create_snippet_actions()
        self._snippet_message = ':'

    def disable_snippet_mode(self):
        self.status_message = ''
        # Reestablishes the shortcuts.
        self._remove_actions()
        self._set_default_shortcuts()
        self._create_actions()
        info("Snippet mode disabled.")

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
            # Default view model kwargs.
            kwargs.update(self._view_model_kwargs(name))
            # View model parameters from settings.
            vm_class = self._view_creator._widget_classes[name]
            kwargs.update(vm_class.get_params(self.settings))
            # debug("Create {} with {}.".format(name, kwargs))
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
        dw = self._dock.add_view(view, title=title, position=position)

        # Dock widget close event.
        @dw.connect_
        def on_close_widget():
            self._view_creator.remove(item)
            self.emit('close_view', item)

        # Make sure the callback above is called when the dock widget
        # is closed directly.
        # View model close event.
        @item.connect
        def on_close_view(e=None):
            dw.close()

        # Call the user callback function.
        if 'on_view_open' in self.settings:
            self.settings['on_view_open'](self, item)

        self.emit('add_view', item)

    def get_views(self, *names):
        """Return the list of views of a given type."""
        return self._view_creator.get(*names)

    def connect_views(self, name_0, name_1):
        """Decorator for a function called on every pair of views of a
        given type."""
        def wrap(func):
            for view_0 in self.get_views(name_0):
                for view_1 in self.get_views(name_1):
                    func(view_0, view_1)
        return wrap

    @property
    def views(self):
        """List of all open views."""
        return self.get_views()

    def view_count(self):
        """Number of views of each type."""
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
        self._dock.close()

    def exit(self):
        """Close the GUI."""
        self.close()
