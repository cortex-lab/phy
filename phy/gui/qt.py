# -*- coding: utf-8 -*-

"""Qt utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from contextlib import contextmanager
from functools import wraps
import logging
import sys

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

from PyQt4.QtCore import (Qt, QByteArray, QMetaObject, QObject,  # noqa
                          QVariant, QEventLoop, QTimer,
                          pyqtSignal, pyqtSlot, QSize, QUrl)
try:
    from PyQt4.QtCore import QPyNullVariant  # noqa
except:  # pragma: no cover
    QPyNullVariant = None
try:
    from PyQt4.QtCore import QString  # noqa
except:  # pragma: no cover
    QString = None
from PyQt4.QtGui import (QKeySequence, QAction, QStatusBar,  # noqa
                         QMainWindow, QDockWidget, QWidget,
                         QMessageBox, QApplication, QMenuBar,
                         QInputDialog,
                         )
from PyQt4.QtWebKit import QWebView, QWebPage, QWebSettings   # noqa


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _button_enum_from_name(name):
    return getattr(QMessageBox, name.capitalize())


def _button_name_from_enum(enum):
    names = dir(QMessageBox)
    for name in names:
        if getattr(QMessageBox, name) == enum:
            return name.lower()


def _prompt(message, buttons=('yes', 'no'), title='Question'):
    buttons = [(button, _button_enum_from_name(button)) for button in buttons]
    arg_buttons = 0
    for (_, button) in buttons:
        arg_buttons |= button
    box = QMessageBox()
    box.setWindowTitle(title)
    box.setText(message)
    box.setStandardButtons(arg_buttons)
    box.setDefaultButton(buttons[0][1])
    return box


def _show_box(box):  # pragma: no cover
    return _button_name_from_enum(box.exec_())


def _input_dialog(title, sentence):
    return QInputDialog.getText(None, title, sentence)  # pragma: no cover


@contextmanager
def _wait_signal(signal, timeout=None):
    """Block loop until signal emitted, or timeout (ms) elapses."""
    # http://jdreaver.com/posts/2014-07-03-waiting-for-signals-pyside-pyqt.html
    loop = QEventLoop()
    signal.connect(loop.quit)

    yield

    if timeout is not None:
        QTimer.singleShot(timeout, loop.quit)
    loop.exec_()


# -----------------------------------------------------------------------------
# Qt app
# -----------------------------------------------------------------------------

def require_qt(func):
    """Specify that a function requires a Qt application.

    Use this decorator to specify that a function needs a running
    Qt application before it can run. An error is raised if that is not
    the case.

    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        return func(*args, **kwargs)
    return wrapped


# Global variable with the current Qt application.
QT_APP = None


def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QApplication(sys.argv)
    return QT_APP


@require_qt
def run_app():  # pragma: no cover
    """Run the Qt application."""
    global QT_APP
    return QT_APP.exit(QT_APP.exec_())


# -----------------------------------------------------------------------------
# Testing utilities
# -----------------------------------------------------------------------------

def _debug_trace():  # pragma: no cover
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()
