# -*- coding: utf-8 -*-

"""Qt dock window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

from six import string_types, PY3

from .qt import QtCore, QtGui
from ..utils.event import EventEmitter

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Qt utilities
# -----------------------------------------------------------------------------

def _title(widget):
    return str(widget.windowTitle()).lower()


def _show_shortcut(shortcut):
    if isinstance(shortcut, string_types):
        return shortcut
    elif isinstance(shortcut, (tuple, list)):
        return ', '.join(shortcut)


def _show_shortcuts(shortcuts, name=None):
    name = name or ''
    print()
    if name:
        name = ' for ' + name
    print('Keyboard shortcuts' + name)
    for name in sorted(shortcuts):
        print('{0:<40}: {1:s}'.format(name, _show_shortcut(shortcuts[name])))
    print()


# -----------------------------------------------------------------------------
# Snippet parsing utilities
# -----------------------------------------------------------------------------

def _parse_arg(s):
    """Parse a number or string."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_list(s):
    """Parse a comma-separated list of values (strings or numbers)."""
    # Range: 'x-y'
    if '-' in s:
        m, M = map(_parse_arg, s.split('-'))
        return tuple(range(m, M + 1))
    # List of ids: 'x,y,z'
    elif ',' in s:
        return tuple(map(_parse_arg, s.split(',')))
    else:
        return _parse_arg(s)


def _parse_snippet(s):
    """Parse an entire snippet command."""
    return list(map(_parse_list, s.split(' ')))


# -----------------------------------------------------------------------------
# Companion class
# -----------------------------------------------------------------------------

class Actions(EventEmitter):
    """Handle GUI actions."""
    def __init__(self):
        super(Actions, self).__init__()
        self._dock = None
        self._actions = {}

    def reset(self):
        """Reset the actions.

        All actions should be registered here, as follows:

        ```python
        @actions.connect
        def on_reset():
            actions.add(...)
            actions.add(...)
            ...
        ```

        """
        self.remove_all()
        self.emit('reset')

    def attach(self, dock):
        """Attach a DockWindow."""
        self._dock = dock

        # Default exit action.
        @self.shortcut('ctrl+q')
        def exit():
            dock.close()

    def add(self, name, callback=None, shortcut=None, alias=None,
            checkable=False, checked=False):
        """Add an action with a keyboard shortcut."""
        # TODO: add menu_name option and create menu bar
        # Get the alias from the character after & if it exists.
        if alias is None:
            alias = name[name.index('&') + 1] if '&' in name else name
        name = name.replace('&', '')
        if name in self._actions:
            return
        action = QtGui.QAction(name, self._dock)
        action.triggered.connect(callback)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if shortcut:
            if not isinstance(shortcut, (tuple, list)):
                shortcut = [shortcut]
            for key in shortcut:
                action.setShortcut(key)
        # HACK: add the shortcut string to the QAction object so that
        # it can be shown in show_shortcuts(). I don't manage to recover
        # the key sequence string from a QAction using Qt.
        action._shortcut_string = shortcut or ''
        # The alias is used in snippets.
        action._alias = alias
        if self._dock:
            self._dock.addAction(action)
        self._actions[name] = action
        logger.debug("Add action `%s`, alias `%s`, shortcut %s.",
                     name, alias, shortcut or '')
        if callback:
            setattr(self, name, callback)
        return action

    def get_name(self, alias_or_name):
        """Return an action name from its alias or name."""
        for name, action in self._actions.items():
            if alias_or_name in (action._alias, name):
                return name

    def remove(self, name):
        """Remove an action."""
        if self._dock:
            self._dock.removeAction(self._actions[name])
        del self._actions[name]
        delattr(self, name)

    def remove_all(self):
        """Remove all actions."""
        names = sorted(self._actions.keys())
        for name in names:
            self.remove(name)

    @property
    def shortcuts(self):
        """A dictionary of action shortcuts."""
        return {name: action._shortcut_string
                for name, action in self._actions.items()}

    def show_shortcuts(self):
        """Print all shortcuts."""
        _show_shortcuts(self.shortcuts,
                        self._dock.title() if self._dock else None)

    def shortcut(self, key=None, name=None, **kwargs):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add(name or func.__name__, shortcut=key,
                     callback=func, **kwargs)
        return wrap


class Snippets(object):
    # HACK: Unicode characters do not appear to work on Python 2
    cursor = '\u200A\u258C' if PY3 else ''

    # Allowed characters in snippet mode.
    # A Qt shortcut will be created for every character.
    _snippet_chars = ("abcdefghijklmnopqrstuvwxyz0123456789"
                      " ,.;:?!_-+~=*/\\(){}[]")

    def __init__(self):
        self._dock = None
        self._cmd = ''  # only used when there is no dock attached

    def attach(self, dock, actions):
        self._dock = dock
        self._actions = actions

        # Register snippet mode shortcut.
        @actions.connect
        def on_reset():
            @actions.shortcut(':')
            def enable_snippet_mode():
                self.mode_on()

    @property
    def command(self):
        """This is used to write a snippet message in the status bar.

        A cursor is appended at the end.

        """
        msg = self._dock.status_message if self._dock else self._cmd
        n = len(msg)
        n_cur = len(self.cursor)
        return msg[:n - n_cur]

    @command.setter
    def command(self, value):
        value += self.cursor
        if not self._dock:
            self._cmd = value
        else:
            self._dock.status_message = value

    def _backspace(self):
        """Erase the last character in the snippet command."""
        if self.command == ':':
            return
        self.command = self.command[:-1]

    def _enter(self):
        """Disable the snippet mode and execute the command."""
        command = self.command
        self.disable_snippet_mode()
        self.run(command)

    def _create_snippet_actions(self):
        """Delete all existing actions, and add mock ones for snippet
        keystrokes.

        Used to enable snippet mode.

        """
        self._actions.remove_all()

        # One action per allowed character.
        for i, char in enumerate(self._snippet_chars):

            def _make_func(char):
                def callback():
                    self.command += char
                return callback

            self._actions.add('snippet_{}'.format(i), shortcut=char,
                              callback=_make_func(char))

        self._actions.add('snippet_backspace', shortcut='backspace',
                          callback=self._backspace)
        self._actions.add('snippet_activate', shortcut=('enter', 'return'),
                          callback=self._enter)
        self._actions.add('snippet_disable', shortcut='escape',
                          callback=self.disable_snippet_mode)

    def run(self, snippet):
        """Executes a snippet command.

        May be overridden.

        """
        assert snippet[0] == ':'
        snippet = snippet[1:]
        snippet_args = _parse_snippet(snippet)
        alias = snippet_args[0]
        name = self._actions.get_name(alias)
        if name is None:
            logger.info("The snippet `%s` could not be found.", alias)
            return
        func = getattr(self._actions, name)
        try:
            logger.info("Processing snippet `%s`.", snippet)
            func(*snippet_args[1:])
        except Exception as e:
            logger.warn("Error when executing snippet: %s.", str(e))

    def mode_on(self):
        logger.info("Snippet mode enabled, press `escape` to leave this mode.")
        # Remove all existing actions, and replace them by snippet keystroke
        # actions.
        self._create_snippet_actions()
        self.command = ':'

    def mode_off(self):
        if self._dock:
            self._dock.status_message = ''
        # Reestablishes the shortcuts.
        self._actions.reset()
        logger.info("Snippet mode disabled.")


# -----------------------------------------------------------------------------
# Qt windows
# -----------------------------------------------------------------------------

class DockWidget(QtGui.QDockWidget):
    """A QDockWidget that can emit events."""
    def __init__(self, *args, **kwargs):
        super(DockWidget, self).__init__(*args, **kwargs)
        self._event = EventEmitter()

    def emit(self, *args, **kwargs):
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)


class DockWindow(QtGui.QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets.

    `DockWindow` derives from `QMainWindow`.

    Events
    ------

    close_gui
    show_gui

    Note
    ----

    Use `connect_()`, not `connect()`, because of a name conflict with Qt.

    """
    def __init__(self,
                 position=None,
                 size=None,
                 title=None,
                 ):
        super(DockWindow, self).__init__()
        if title is None:
            title = 'phy'
        self.setWindowTitle(title)
        if position is not None:
            self.move(position[0], position[1])
        if size is not None:
            self.resize(QtCore.QSize(size[0], size[1]))
        self.setObjectName(title)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.setDockOptions(QtGui.QMainWindow.AllowTabbedDocks |
                            QtGui.QMainWindow.AllowNestedDocks |
                            QtGui.QMainWindow.AnimatedDocks
                            )
        # We can derive from EventEmitter because of a conflict with connect.
        self._event = EventEmitter()

        self._status_bar = QtGui.QStatusBar()
        self.setStatusBar(self._status_bar)

    # Events
    # -------------------------------------------------------------------------

    def emit(self, *args, **kwargs):
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def unconnect_(self, *args, **kwargs):
        self._event.unconnect(*args, **kwargs)

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        res = self.emit('close_gui')
        # Discard the close event if False is returned by one of the callback
        # functions.
        if False in res:  # pragma: no cover
            e.ignore()
            return
        super(DockWindow, self).closeEvent(e)

    def show(self):
        """Show the window."""
        self.emit('show_gui')
        super(DockWindow, self).show()

    # Views
    # -------------------------------------------------------------------------

    def add_view(self,
                 view,
                 title='view',
                 position=None,
                 closable=True,
                 floatable=True,
                 floating=None,
                 # parent=None,  # object to pass in the raised events
                 **kwargs):
        """Add a widget to the main window."""

        try:
            from vispy.app import Canvas
            if isinstance(view, Canvas):
                view = view.native
        except ImportError:  # pragma: no cover
            pass

        # Create the dock widget.
        dockwidget = DockWidget(self)
        dockwidget.setObjectName(title)
        dockwidget.setWindowTitle(title)
        dockwidget.setWidget(view)

        # Set dock widget options.
        options = QtGui.QDockWidget.DockWidgetMovable
        if closable:
            options = options | QtGui.QDockWidget.DockWidgetClosable
        if floatable:
            options = options | QtGui.QDockWidget.DockWidgetFloatable

        dockwidget.setFeatures(options)
        dockwidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea |
                                   QtCore.Qt.RightDockWidgetArea |
                                   QtCore.Qt.TopDockWidgetArea |
                                   QtCore.Qt.BottomDockWidgetArea
                                   )

        q_position = {
            'left': QtCore.Qt.LeftDockWidgetArea,
            'right': QtCore.Qt.RightDockWidgetArea,
            'top': QtCore.Qt.TopDockWidgetArea,
            'bottom': QtCore.Qt.BottomDockWidgetArea,
        }[position or 'right']
        self.addDockWidget(q_position, dockwidget)
        if floating is not None:
            dockwidget.setFloating(floating)
        dockwidget.show()
        return dockwidget

    def list_views(self, title='', is_visible=True):
        """List all views which title start with a given string."""
        title = title.lower()
        children = self.findChildren(QtGui.QWidget)
        return [child for child in children
                if isinstance(child, QtGui.QDockWidget) and
                _title(child).startswith(title) and
                (child.isVisible() if is_visible else True) and
                child.width() >= 10 and
                child.height() >= 10
                ]

    def view_count(self):
        """Return the number of opened views."""
        views = self.list_views()
        counts = defaultdict(lambda: 0)
        for view in views:
            counts[_title(view)] += 1
        return dict(counts)

    # Status bar
    # -------------------------------------------------------------------------

    @property
    def status_message(self):
        """The message in the status bar."""
        return str(self._status_bar.currentMessage())

    @status_message.setter
    def status_message(self, value):
        self._status_bar.showMessage(str(value))

    # State
    # -------------------------------------------------------------------------

    def save_geometry_state(self):
        """Return picklable geometry and state of the window and docks.

        This function can be called in `on_close()`.

        """
        return {
            'geometry': self.saveGeometry(),
            'state': self.saveState(),
            'view_count': self.view_count(),
        }

    def restore_geometry_state(self, gs):
        """Restore the position of the main window and the docks.

        The dock widgets need to be recreated first.

        This function can be called in `on_show()`.

        """
        if gs.get('geometry', None):
            self.restoreGeometry((gs['geometry']))
        if gs.get('state', None):
            self.restoreState((gs['state']))
