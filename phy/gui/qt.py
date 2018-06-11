# -*- coding: utf-8 -*-

"""Qt utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from contextlib import contextmanager
from functools import wraps
import logging
import os.path as op
import sys
from timeit import default_timer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

# BUG FIX on Ubuntu, otherwise the canvas is all black:
# https://riverbankcomputing.com/pipermail/pyqt/2014-January/033681.html
from OpenGL import GL  # noqa

from PyQt5.QtCore import (Qt, QByteArray, QMetaObject, QObject,  # noqa
                          QVariant, QEventLoop, QTimer, QPoint, QTimer,
                          pyqtSignal, pyqtSlot, QSize, QUrl,
                          )
from PyQt5.QtGui import QKeySequence, QColor  # noqa
from PyQt5.QtWebEngineWidgets import (QWebEngineView,  # noqa
                                      QWebEnginePage,
                                      # QWebSettings,
                                      )
from PyQt5.QtWebChannel import QWebChannel  # noqa
from PyQt5.QtWidgets import (QAction, QStatusBar,  # noqa
                             QMainWindow, QDockWidget, QWidget,
                             QMessageBox, QApplication, QMenuBar,
                             QInputDialog,
                             )

# Enable high DPI support.
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)


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
        QTimer.singleShot(timeout, loop.quit)  # pragma: no cover
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


def block(until_true):
    t0 = default_timer()
    timeout = .2
    while not until_true() and (default_timer() - t0 < timeout):
        app = QApplication.instance()
        app.processEvents(QEventLoop.ExcludeUserInputEvents |
                          QEventLoop.ExcludeSocketNotifiers |
                          QEventLoop.WaitForMoreEvents)


def _abs_path(rel_path):
    static_dir = op.join(op.abspath(op.dirname(__file__)), 'static/')
    return op.join(static_dir, rel_path)


class WebPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, source):
        super(WebPage, self).javaScriptConsoleMessage(level, msg, line, source)
        msg = "[JS:L%02d] %s" % (line, msg)
        f = (logger.debug, logger.warn, logger.error)[level]
        if level >= 2:
            raise RuntimeError(msg)
        f(msg)


class WebView(QWebEngineView):
    def __init__(self, *args):
        super(WebView, self).__init__(*args)
        self._page = WebPage()
        self.setPage(self._page)
        self.move(100, 100)
        self.resize(400, 400)

    def set_html_sync(self, html):
        self.html = None
        self.loadFinished.connect(self._loadFinished)
        static_dir = op.join(op.realpath(op.dirname(__file__)), 'static/')
        base_url = QUrl().fromLocalFile(static_dir)
        self.page().setHtml(html, base_url)
        block(lambda: self.html is not None)

    def _callable(self, data):
        self.html = data

    def _loadFinished(self, result):
        self.page().toHtml(self._callable)


# -----------------------------------------------------------------------------
# Testing utilities
# -----------------------------------------------------------------------------

def _debug_trace():  # pragma: no cover
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt5.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()
