# -*- coding: utf-8 -*-

"""Qt dock window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict

from .qt import QtCore, QtGui
from ..utils.event import EventEmitter
from ..ext.six import u


# -----------------------------------------------------------------------------
# Qt utilities
# -----------------------------------------------------------------------------

def _title(widget):
    return str(widget.windowTitle()).lower()


def _widget(dock_widget):
    """Return a Qt or VisPy widget from a dock widget."""
    widget = dock_widget.widget()
    if hasattr(widget, '_vispy_canvas'):
        return widget._vispy_canvas
    else:
        return widget


# -----------------------------------------------------------------------------
# Qt windows
# -----------------------------------------------------------------------------

class DockWindow(QtGui.QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets.

    Events
    ------

    close_gui
    show_gui
    keystroke

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
        self._actions = {}
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

    def keyReleaseEvent(self, e):
        self.emit('keystroke', e.key(), e.text())
        return super(DockWindow, self).keyReleaseEvent(e)

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
        if False in res:
            e.ignore()
            return
        super(DockWindow, self).closeEvent(e)

    def show(self):
        """Show the window."""
        self.emit('show_gui')
        super(DockWindow, self).show()

    # Actions
    # -------------------------------------------------------------------------

    def add_action(self,
                   name,
                   callback=None,
                   shortcut=None,
                   checkable=False,
                   checked=False,
                   ):
        """Add an action with a keyboard shortcut."""
        if name in self._actions:
            return
        action = QtGui.QAction(name, self)
        action.triggered.connect(callback)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if shortcut:
            if not isinstance(shortcut, (tuple, list)):
                shortcut = [shortcut]
            for key in shortcut:
                action.setShortcut(key)
        self.addAction(action)
        self._actions[name] = action
        if callback:
            setattr(self, name, callback)
        return action

    def remove_action(self, name):
        """Remove an action."""
        self.removeAction(self._actions[name])
        del self._actions[name]
        delattr(self, name)

    def remove_actions(self):
        """Remove all actions."""
        names = sorted(self._actions.keys())
        for name in names:
            self.remove_action(name)

    def shortcut(self, name, key=None):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add_action(name, shortcut=key, callback=func)
        return wrap

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
        except ImportError:
            pass

        class DockWidget(QtGui.QDockWidget):
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

    def view_count(self, is_visible=True):
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
        return u(self._status_bar.currentMessage())

    @status_message.setter
    def status_message(self, value):
        self._status_bar.showMessage(u(value))

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
