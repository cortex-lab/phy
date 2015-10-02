# -*- coding: utf-8 -*-

"""Qt dock window."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

from .qt import QtCore, QtGui
from phy.utils.event import EventEmitter
from phy.utils.plugin import get_plugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GUI main window
# -----------------------------------------------------------------------------

def _title(widget):
    return str(widget.windowTitle()).lower()


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


class GUI(QtGui.QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets.

    `GUI` derives from `QMainWindow`.

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
        super(GUI, self).__init__()
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

    def attach(self, plugin_name):
        plugin = get_plugin(name)
        plugin.attach_gui(self)

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
        super(GUI, self).closeEvent(e)

    def show(self):
        """Show the window."""
        self.emit('show_gui')
        super(GUI, self).show()

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

        # Create the gui widget.
        dockwidget = DockWidget(self)
        dockwidget.setObjectName(title)
        dockwidget.setWindowTitle(title)
        dockwidget.setWidget(view)

        # Set gui widget options.
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

        The gui widgets need to be recreated first.

        This function can be called in `on_show()`.

        """
        if gs.get('geometry', None):
            self.restoreGeometry((gs['geometry']))
        if gs.get('state', None):
            self.restoreState((gs['state']))
