# -*- coding: utf-8 -*-

"""Qt dock window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict

from .qt import QtCore, QtGui
from ..utils.event import EventEmitter


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
# Dock main window
# -----------------------------------------------------------------------------

class DockWindow(QtGui.QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets.

    Events
    ------

    close_widget
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

    # Events
    # -------------------------------------------------------------------------

    def emit(self, *args, **kwargs):
        self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_gui')
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
        return action

    def shortcut(self, name, key=None):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add_action(name, shortcut=key, callback=func)
            setattr(self, name, func)
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
                 parent=None,  # object to pass in the raised events
                 **kwargs):
        """Add a widget to the main window."""

        try:
            from vispy.app import Canvas
            if isinstance(view, Canvas):
                view = view.native
        except ImportError:
            pass

        parent = self

        class DockWidget(QtGui.QDockWidget):
            def closeEvent(self, e):
                """Qt slot when the window is closed."""
                parent.emit('close_widget', parent or view)
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

    def connect_views(self, name_0, name_1):
        """Decorator for a function that accepts any pair of views.

        This is used to connect any view of type `name_0` to any other view
        of type `name_1`.

        """
        def _make_func(func):
            for widget_0 in self.list_views(name_0, is_visible=False):
                for widget_1 in self.list_views(name_1, is_visible=False):
                    view_0 = _widget(widget_0)
                    view_1 = _widget(widget_1)
                    func(view_0, view_1)
        return _make_func

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
        self.restoreGeometry((gs['geometry']))
        self.restoreState((gs['state']))
