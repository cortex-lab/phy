# -*- coding: utf-8 -*-

"""Qt utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import contextlib
import logging
import os
import sys

from ..utils._misc import _is_interactive

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

_PYQT = False
try:
    from PyQt4 import QtCore, QtGui, QtWebKit  # noqa
    Qt = QtCore.Qt
    _PYQT = True
except ImportError:  # pragma: no cover
    try:
        from PyQt5 import QtCore, QtGui, QtWebKit  # noqa
        _PYQT = True
    except ImportError:
        pass


def _check_qt():  # pragma: no cover
    if not _PYQT:
        logger.warn("PyQt is not available.")
        return False
    return True


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _button_enum_from_name(name):
    return getattr(QtGui.QMessageBox, name.capitalize())


def _button_name_from_enum(enum):
    names = dir(QtGui.QMessageBox)
    for name in names:
        if getattr(QtGui.QMessageBox, name) == enum:
            return name.lower()


def _prompt(message, buttons=('yes', 'no'), title='Question'):
    buttons = [(button, _button_enum_from_name(button)) for button in buttons]
    arg_buttons = 0
    for (_, button) in buttons:
        arg_buttons |= button
    box = QtGui.QMessageBox()
    box.setWindowTitle(title)
    box.setText(message)
    box.setStandardButtons(arg_buttons)
    box.setDefaultButton(buttons[0][1])
    return box


def _show_box(box):  # pragma: no cover
    return _button_name_from_enum(box.exec_())


# -----------------------------------------------------------------------------
# Event loop integration with IPython
# -----------------------------------------------------------------------------

_APP = None
_APP_RUNNING = False


def _try_enable_ipython_qt():  # pragma: no cover
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


def enable_qt():  # pragma: no cover
    if not _check_qt():
        return
    try:
        from IPython import get_ipython
        ip = get_ipython()
        ip.enable_gui('qt')
        global _APP_RUNNING
        _APP_RUNNING = True
        logger.info("Qt event loop activated.")
    except:
        logger.warn("Qt event loop not activated.")


# -----------------------------------------------------------------------------
# Qt app
# -----------------------------------------------------------------------------

def start_qt_app():  # pragma: no cover
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


def run_qt_app():  # pragma: no cover
    """Start the Qt application's event loop."""
    global _APP_RUNNING
    if not _check_qt():
        return
    if _APP is not None and not _APP_RUNNING:
        _APP_RUNNING = True
        _APP.exec_()
    if not _is_interactive():
        _APP_RUNNING = False


# -----------------------------------------------------------------------------
# Testing utilities
# -----------------------------------------------------------------------------

_MAX_ITER = 100
_DELAY = max(0, float(os.environ.get('PHY_EVENT_LOOP_DELAY', .1)))


def _debug_trace():  # pragma: no cover
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()
