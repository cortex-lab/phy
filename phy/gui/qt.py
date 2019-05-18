# -*- coding: utf-8 -*-

"""Qt utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from contextlib import contextmanager
from functools import wraps, partial
import logging
from pathlib import Path
import sys
from timeit import default_timer
import traceback

from phylib.utils.testing import _in_travis

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

# BUG FIX on Ubuntu, otherwise the canvas is all black:
# https://riverbankcomputing.com/pipermail/pyqt/2014-January/033681.html
from OpenGL import GL  # noqa

from PyQt5.QtCore import (Qt, QByteArray, QMetaObject, QObject,  # noqa
                          QVariant, QEventLoop, QTimer, QPoint, QTimer,
                          QThreadPool, QRunnable,
                          pyqtSignal, pyqtSlot, QSize, QUrl,
                          QEvent, QCoreApplication,
                          qInstallMessageHandler,
                          )
from PyQt5.QtGui import (  # noqa
    QKeySequence, QColor, QMouseEvent, QGuiApplication,
    QWindow, QOpenGLWindow)
from PyQt5.QtWebEngineWidgets import (QWebEngineView,  # noqa
                                      QWebEnginePage,
                                      # QWebSettings,
                                      )
from PyQt5.QtWebChannel import QWebChannel  # noqa
from PyQt5.QtWidgets import (QAction, QStatusBar,  # noqa
                             QMainWindow, QDockWidget, QWidget,
                             QMessageBox, QApplication, QMenuBar,
                             QInputDialog, QOpenGLWidget
                             )

# Enable high DPI support.
# BUG: uncommenting this create scaling bugs on high DPI screens
# on Ubuntu.
#QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)


# -----------------------------------------------------------------------------
# Testing functions: mock dialogs in automated tests
# -----------------------------------------------------------------------------

_MOCK = None


def mockable(f):
    def wrapped(*args, **kwargs):
        if _MOCK is not None:
            return _MOCK
        return f(*args, **kwargs)
    return wrapped


@contextmanager
def mock_dialogs(result):
    """All mockable functions just return some output instead of prompting a dialog.

    The returned result should not be None.

    """
    assert result is not None
    globals()['_MOCK'] = result
    yield
    globals()['_MOCK'] = None


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

@mockable
def _button_enum_from_name(name):
    return getattr(QMessageBox, name.capitalize())


@mockable
def _button_name_from_enum(enum):
    names = dir(QMessageBox)
    for name in names:
        if getattr(QMessageBox, name) == enum:
            return name.lower()


@mockable
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


@mockable
def _show_box(box):  # pragma: no cover
    return _button_name_from_enum(box.exec_())


@mockable
def _input_dialog(title, sentence, text=None):
    return QInputDialog.getText(None, title, sentence,
                                text=text)  # pragma: no cover


def _screen_size():
    screen = QGuiApplication.primaryScreen()
    geometry = screen.geometry()
    return (geometry.width(), geometry.height())


def _is_high_dpi():
    return _screen_size()[0] > 3000


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


def set_busy(busy):
    app = create_app()
    if busy:
        app.setOverrideCursor(Qt.WaitCursor)
    else:
        app.restoreOverrideCursor()
        app.restoreOverrideCursor()


@contextmanager
def busy_cursor():
    """Context manager displaying a busy cursor during a long command."""
    set_busy(True)
    yield
    set_busy(False)


def _block(until_true, timeout=None):
    if until_true():
        return
    t0 = default_timer()
    timeout = timeout or (2 if not _in_travis() else 5)

    while not until_true() and (default_timer() - t0 < timeout):
        app = QApplication.instance()
        app.processEvents(QEventLoop.AllEvents,
                          int(timeout * 1000))
    if not until_true():
        logger.error("Timeout in _block().")
        raise RuntimeError("Timeout in _block().")


def _wait(ms):
    from PyQt5 import QtTest
    QtTest.QTest.qWait(ms)


# -----------------------------------------------------------------------------
# Threading
# -----------------------------------------------------------------------------

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):  # pragma: no cover
        # Bug with coverage, which doesn't recognize these lines as
        # called when they are called from a different thread.
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


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


def _abs_path(rel_path):
    return Path(__file__).parent / 'static' / rel_path


class WebPage(QWebEnginePage):
    _raise_on_javascript_error = False

    def javaScriptConsoleMessage(self, level, msg, line, source):
        super(WebPage, self).javaScriptConsoleMessage(level, msg, line, source)
        msg = "[JS:L%02d] %s" % (line, msg)
        f = (partial(logger.log, 5), logger.warning, logger.error)[level]
        if self._raise_on_javascript_error and level >= 2:
            raise RuntimeError(msg)
        f(msg)


class WebView(QWebEngineView):
    def __init__(self, *args):
        super(WebView, self).__init__(*args)
        self.html = None
        assert isinstance(self.window(), QWidget)
        self._page = WebPage(self)
        self.setPage(self._page)
        self.move(100, 100)
        self.resize(400, 400)

    def set_html(self, html, callback=None):
        self._callback = callback
        self.loadFinished.connect(self._loadFinished)
        static_dir = str(Path(__file__).parent / 'static') + '/'
        base_url = QUrl().fromLocalFile(static_dir)
        self.page().setHtml(html, base_url)

    def _callable(self, data):
        self.html = data
        if self._callback:
            self._callback(self.html)

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
