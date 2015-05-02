# -*- coding: utf-8 -*-

"""Qt dock window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QMainWindow
from vispy import app

from ._misc import _is_interactive


# -----------------------------------------------------------------------------
# Dock main window
# -----------------------------------------------------------------------------

class DockWindow(QMainWindow):
    def __init__(self,
                 position=(1100, 200),
                 size=(800, 600),
                 title=None,
                 ):
        super(DockWindow, self).__init__()
        if title is None:
            title = 'phy'
        self.setWindowTitle(title)
        self.move(position[0], position[1])
        self.resize(QtCore.QSize(size[0], size[1]))
        self.setObjectName("MainWindow")
        QtCore.QMetaObject.connectSlotsByName(self)
        self.setDockOptions(QtGui.QMainWindow.AllowTabbedDocks |
                            QtGui.QMainWindow.AllowNestedDocks |
                            QtGui.QMainWindow.AnimatedDocks
                            )

    def add_action(self,
                   text,
                   callback=None,
                   shortcut=None,
                   checkable=False,
                   checked=False,
                   ):
        """Add an action with a keyboard shortcut."""
        action = QtGui.QAction(text, self)
        action.triggered.connect(callback)
        action.setShortcut(shortcut)
        action.setCheckable(checkable)
        action.setChecked(checked)
        self.addAction(action)
        return action

    def add_view(self,
                 view,
                 title='view',
                 position='right',
                 closable=True,
                 floatable=True,
                 floating=None,
                 **kwargs):
        """Add a widget to the main window."""

        if isinstance(view, app.Canvas):
            view = view.native

        # Create the dock widget.
        dockwidget = QtGui.QDockWidget(self)
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
        }[position]
        self.addDockWidget(q_position, dockwidget)
        if floating is not None:
            dockwidget.setFloating(floating)
        dockwidget.show()
        return dockwidget

    def list_views(self, title=''):
        """List all views which title start with a given string."""
        children = self.findChildren(QtGui.QWidget)
        return [child for child in children
                if str(child.windowTitle()).startswith(title)]

    def shortcut(self, text, key):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add_action(text, shortcut=key, callback=func)
        return wrap


# -----------------------------------------------------------------------------
# Qt app and event loop integration with IPython
# -----------------------------------------------------------------------------

_APP = None


def _close_qt_after(window, duration):
    """Close a Qt window after a given duration."""
    def callback():
        window.close()
    QtCore.QTimer.singleShot(int(1000 * duration), callback)


def _try_enable_ipython_qt():
    """Try to enable IPython Qt event loop integration.

    Returns True in the following cases:

    * python -i test.py
    * ipython -i test.py
    * ipython and %run test.py

    Returns False in the following cases:

    * python test.py
    * ipython test.py

    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        return False
    if not _is_interactive():
        return False
    if ip:
        qt_app = ip.enable_gui('qt')
        if qt_app:
            return qt_app
    return False


def start_qt_app():
    """Start a Qt application if necessary.

    If a new Qt application is created, this function returns it.
    If no new application is created, the function returns None.

    """
    # Only start a Qt application if there is no
    # IPython event loop integration.
    global _APP
    if _try_enable_ipython_qt():
        return
    if QtGui.QApplication.instance():
        _APP = QtGui.QApplication.instance()
        return
    if _APP:
        return
    app.use_app("pyqt4")
    _APP = QtGui.QApplication(sys.argv)
    return _APP


def run_qt_app():
    """Start the Qt application's event loop."""
    if _APP is not None:
        _APP.exec_()
