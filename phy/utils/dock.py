# -*- coding: utf-8 -*-

"""Qt dock window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys
import contextlib
from collections import defaultdict

from ._misc import _is_interactive
from .logging import info, warn


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

_PYQT = False
try:
    from PyQt4 import QtCore, QtGui
    from PyQt4.QtGui import QMainWindow
    _PYQT = True
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui
        from PyQt5.QtGui import QMainWindow
        _PYQT = True
    except ImportError:
        pass


def _check_qt():
    if not _PYQT:
        warn("PyQt is not available.")
        return False
    return True


if not _check_qt():
    QMainWindow = object  # noqa


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _title(widget):
    return str(widget.windowTitle()).lower()


def _create_web_view(html=None):
    from PyQt4.QtWebKit import QWebView
    view = QWebView()
    if html:
        view.setHtml(html)
    return view


def _widget(dock_widget):
    """Return a Qt or VisPy widget from a dock widget."""
    widget = dock_widget.widget()
    if hasattr(widget, '_vispy_canvas'):
        return widget._vispy_canvas
    else:
        return widget


def _prompt(parent, message, buttons=('yes', 'no'), title='Question'):
    buttons = [(button, getattr(QtGui.QMessageBox, button.capitalize()))
               for button in buttons]
    arg_buttons = 0
    for (_, button) in buttons:
        arg_buttons |= button
    reply = QtGui.QMessageBox.question(parent,
                                       title,
                                       message,
                                       arg_buttons,
                                       buttons[0][1],
                                       )
    for name, button in buttons:
        if reply == button:
            return name


# -----------------------------------------------------------------------------
# Qt app and event loop integration with IPython
# -----------------------------------------------------------------------------

_APP = None
_APP_RUNNING = False


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
        ip.enable_gui('qt')
        global _APP_RUNNING
        _APP_RUNNING = True
        return True
    return False


def enable_qt():
    if not _check_qt():
        return
    try:
        from IPython import get_ipython
        ip = get_ipython()
        ip.enable_gui('qt')
        global _APP_RUNNING
        _APP_RUNNING = True
        info("Qt event loop activated.")
    except:
        warn("Qt event loop not activated.")


def start_qt_app():
    """Start a Qt application if necessary.

    If a new Qt application is created, this function returns it.
    If no new application is created, the function returns None.

    """
    # Only start a Qt application if there is no
    # IPython event loop integration.
    if not _check_qt():
        return
    global _APP
    if _try_enable_ipython_qt():
        return
    try:
        from vispy import app
        app.use_app("pyqt4")
    except ImportError:
        pass
    if QtGui.QApplication.instance():
        _APP = QtGui.QApplication.instance()
        return
    if _APP:
        return
    _APP = QtGui.QApplication(sys.argv)
    return _APP


def run_qt_app():
    """Start the Qt application's event loop."""
    global _APP_RUNNING
    if not _check_qt():
        return
    if _APP is not None and not _APP_RUNNING:
        _APP_RUNNING = True
        _APP.exec_()
    if not _is_interactive():
        _APP_RUNNING = False


@contextlib.contextmanager
def qt_app():
    """Context manager to ensure that a Qt app is running."""
    if not _check_qt():
        return
    app = start_qt_app()
    yield app
    run_qt_app()


# -----------------------------------------------------------------------------
# Dock main window
# -----------------------------------------------------------------------------

class DockWindow(QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets."""
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
        self._on_show = None
        self._on_close = None

    # Events
    # -------------------------------------------------------------------------

    def on_close(self, func):
        """Register a callback function when the window is closed."""
        self._on_close = func

    def on_show(self, func):
        """Register a callback function when the window is shown."""
        self._on_show = func

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        if self._on_close:
            self._on_close()
        super(DockWindow, self).closeEvent(e)

    def show(self):
        """Show the window."""
        if self._on_show:
            self._on_show()
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
                 **kwargs):
        """Add a widget to the main window."""

        try:
            from vispy.app import Canvas
            if isinstance(view, Canvas):
                view = view.native
        except ImportError:
            pass

        # Create the dock widget.
        dockwidget = QtGui.QDockWidget(self)
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

    def view_counts(self):
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
            'view_counts': self.view_counts(),
        }

    def restore_geometry_state(self, gs):
        """Restore the position of the main window and the docks.

        The dock widgets need to be recreated first.

        This function can be called in `on_show()`.

        """
        self.restoreGeometry((gs['geometry']))
        self.restoreState((gs['state']))
