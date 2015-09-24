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

    @property
    def shortcuts(self):
        """A dictionary of action shortcuts."""
        return {name: action._shortcut_string
                for name, action in self._actions.items()}

    def show_shortcuts(self):
        """Print all shortcuts."""
        _show_shortcuts(self.shortcuts,
                        self._dock.title() if self._dock else None)

    def add(self, name, callback=None, shortcut=None,
            checkable=False, checked=False):
        """Add an action with a keyboard shortcut."""
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
        if self._dock:
            self._dock.addAction(action)
        self._actions[name] = action
        if callback:
            setattr(self, name, callback)
        return action

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

    def shortcut(self, key=None, name=None):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add(name or func.__name__, shortcut=key, callback=func)
        return wrap


class Snippets(object):
    # HACK: Unicode characters do not appear to work on Python 2
    cursor = '\u200A\u258C' if PY3 else ''

    # Allowed characters in snippet mode.
    _snippet_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 ._,+*-=:()'

    def __init__(self):
        self._dock = None

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
        n = len(self._dock.status_message)
        n_cur = len(self.cursor)
        return self._dock.status_message[:n - n_cur]

    @command.setter
    def command(self, value):
        self._dock.status_message = value + self.cursor

    def run(self, snippet):
        """Executes a snippet.

        May be overriden.

        """
        assert snippet[0] == ':'
        snippet = snippet[1:]
        split = snippet.split(' ')
        cmd = split[0]
        snippet = snippet[len(cmd):].strip()
        func = self._snippets.get(cmd, None)
        if func is None:
            logger.info("The snippet `%s` could not be found.", cmd)
            return
        try:
            logger.info("Processing snippet `%s`.", cmd)
            func(self, snippet)
        except Exception as e:
            logger.warn("Error when executing snippet `%s`: %s.",
                        cmd, str(e))

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

            self._actions.add('snippet_{}'.format(i),
                              shortcut=char,
                              callback=_make_func(char),
                              )

        def backspace():
            if self.command == ':':
                return
            self.command = self.command[:-1]

        def enter():
            self.run(self.command)
            self.disable_snippet_mode()

        self._actions.add('snippet_backspace',
                          shortcut='backspace',
                          callback=backspace,
                          )
        self._actions.add('snippet_activate',
                          shortcut=('enter', 'return'),
                          callback=enter,
                          )
        self._actions.add('snippet_disable',
                          shortcut='escape',
                          callback=self.disable_snippet_mode,
                          )

    def mode_on(self):
        logger.info("Snippet mode enabled, press `escape` to leave this mode.")
        # Remove all existing actions, and replace them by snippet keystroke
        # actions.
        self._create_snippet_actions()
        self.command = ':'

    def mode_off(self):
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
