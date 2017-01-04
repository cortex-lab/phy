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

from PyQt5.QtCore import (Qt, QByteArray, QMetaObject, QObject,  # noqa
                          QVariant, QEventLoop, QTimer, QPoint, QTimer,
                          pyqtSignal, pyqtSlot, QSize, QUrl,
                          )
from PyQt5.QtGui import QKeySequence  # noqa
from PyQt5.QtWebKit import QWebSettings   # noqa
from PyQt5.QtWebKitWidgets import QWebView, QWebPage   # noqa
from PyQt5.QtWidgets import (QAction, QStatusBar,  # noqa
                             QMainWindow, QDockWidget, QWidget,
                             QMessageBox, QApplication, QMenuBar,
                             QInputDialog,
                             )


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


def _input_dialog(title, sentence, text=None):
    return QInputDialog.getText(None, title, sentence,
                                text=text)  # pragma: no cover


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


@contextmanager
def busy_cursor():
    """Context manager displaying a busy cursor during a long command."""
    create_app().setOverrideCursor(Qt.WaitCursor)
    yield
    create_app().restoreOverrideCursor()


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


class AsyncCaller(object):
    """Call a Python function after a delay."""
    def __init__(self, delay=10):
        self._delay = delay
        self._timer = None

    def _create_timer(self, f):
        self._timer = QTimer()
        self._timer.timeout.connect(f)
        self._timer.setSingleShot(True)

    def set(self, f):
        """Call a function after a delay, unless another function is set
        in the meantime."""
        self.stop()
        self._create_timer(f)
        self.start()

    def start(self):
        """Start the timer and call the function after a delay."""
        if self._timer:
            self._timer.start(self._delay)

    def stop(self):
        """Stop the current timer if there is one and cancel the async call."""
        if self._timer:
            self._timer.stop()
            self._timer.deleteLater()


# -----------------------------------------------------------------------------
# Testing utilities
# -----------------------------------------------------------------------------

def _debug_trace():  # pragma: no cover
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt5.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()
