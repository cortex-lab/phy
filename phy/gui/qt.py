# -*- coding: utf-8 -*-

"""Qt utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os
import sys
import contextlib

from ..ext.six import wraps
from ..utils._misc import _is_interactive
from ..utils.logging import info, warn


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

_PYQT = False
try:
    from PyQt4 import QtCore, QtGui, QtWebKit  # noqa
    from PyQt4.QtGui import QMainWindow
    Qt = QtCore.Qt
    _PYQT = True
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWebKit  # noqa
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


def _set_qt_widget_position_size(widget, position=None, size=None):
    if position is not None:
        widget.moveTo(*position)
    if size is not None:
        widget.resize(*size)


# -----------------------------------------------------------------------------
# Event loop integration with IPython
# -----------------------------------------------------------------------------

_APP = None
_APP_RUNNING = False


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


# -----------------------------------------------------------------------------
# Qt app
# -----------------------------------------------------------------------------

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
# Testing utilities
# -----------------------------------------------------------------------------

def _close_qt_after(window, duration):
    """Close a Qt window after a given duration."""
    def callback():
        window.close()
    QtCore.QTimer.singleShot(int(1000 * duration), callback)


_MAX_ITER = 100
_DELAY = max(0, float(os.environ.get('PHY_EVENT_LOOP_DELAY', .1)))


def _debug_trace():
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def wrap_qt(func):
    """Automatically test a Qt view.

    Used as a decorator on a generator function:

    ```python
    @wrap_qt
    def test_qt():
        view = QtGui.QWidget()
        view...
        yield view
        ...
        yield
        ...
    ```

    The application will run for x seconds between two yields. The delay can be
    specified via the command-line with the PHY_EVENT_LOOP_DELAY env variable.

    Notes
    -----

    * You must explicitely call `show()` and `close()`.
    * Never call `yield` right after a `close()` if no window is still open,
      otherwise the Qt application will close automatically.

    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        with qt_app():
            gen = func(*args, **kwargs)
            for i in range(_MAX_ITER):
                def callback():
                    try:
                        next(gen)
                    except (StopIteration, ValueError):
                        pass
                    except Exception as e:
                        _APP.exit()
                        raise e
                QtCore.QTimer.singleShot(int(1000 * _DELAY * i),
                                         callback)

    return wrapped
