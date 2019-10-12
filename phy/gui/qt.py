# -*- coding: utf-8 -*-

"""Qt utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import logging
from pathlib import Path
import sys
from timeit import default_timer
import traceback

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
    QFontDatabase, QWindow, QOpenGLWindow)
from PyQt5.QtWebEngineWidgets import (QWebEngineView,  # noqa
                                      QWebEnginePage,
                                      # QWebSettings,
                                      )
from PyQt5.QtWebChannel import QWebChannel  # noqa
from PyQt5.QtWidgets import (QAction, QStatusBar,  # noqa
                             QMainWindow, QDockWidget, QWidget,
                             QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QCheckBox,
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
    """Wrap interactive Qt functions that should be mockable in the testing suite."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if _MOCK is not None:
            return _MOCK
        return f(*args, **kwargs)
    return wrapped


@contextmanager
def mock_dialogs(result):
    """A context manager such that all mockable functions called inside just return some output
    instead of prompting a dialog.

    **Important: the returned result should not be None.**

    """
    assert result is not None
    globals()['_MOCK'] = result
    yield
    globals()['_MOCK'] = None


# -----------------------------------------------------------------------------
# Qt app
# -----------------------------------------------------------------------------

# Global variable with the current Qt application.
QT_APP = None


def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QApplication(sys.argv)
    return QT_APP


def require_qt(func):
    """Function decorator to specify that a function requires a Qt application.

    Use this decorator to specify that a function needs a running
    Qt application before it can run. An error is raised if that is not
    the case.

    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not QApplication.instance():  # pragma: no cover
            logger.warning("Creating a Qt application.")
            create_app()
        return func(*args, **kwargs)
    return wrapped


@require_qt
def run_app():  # pragma: no cover
    """Run the Qt application."""
    global QT_APP
    return QT_APP.exit(QT_APP.exec_())


# -----------------------------------------------------------------------------
# Internal utility functions
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


_DEFAULT_TIMEOUT = 5  # in seconds


def _block(until_true, timeout=None):
    """Block until the given function returns True. There is a timeout of a couple of seconds."""
    if until_true():
        return
    t0 = default_timer()
    timeout = timeout or _DEFAULT_TIMEOUT

    while not until_true() and (default_timer() - t0 < timeout):
        app = QApplication.instance()
        app.processEvents(QEventLoop.AllEvents,
                          int(timeout * 1000))
    if not until_true():
        logger.error("Timeout in _block().")
        # NOTE: make sure we remove any busy cursor.
        app.restoreOverrideCursor()
        app.restoreOverrideCursor()
        raise RuntimeError("Timeout in _block().")


def _wait(ms):
    """Wait for a given number of milliseconds, without blocking the GUI."""
    from PyQt5 import QtTest
    QtTest.QTest.qWait(ms)


def _debug_trace():  # pragma: no cover
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt5.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

@mockable
def prompt(message, buttons=('yes', 'no'), title='Question'):
    """Display a dialog with several buttons to confirm or cancel an action.

    Parameters
    ----------

    message : str
        Dialog message.
    buttons : tuple
        Name of the standard buttons to show in the prompt: yes, no, ok, cancel, close, etc.
        See the full list at https://doc.qt.io/qt-5/qmessagebox.html#StandardButton-enum
    title : str
        Dialog title.

    """
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
def message_box(message, title='Message', level=None):  # pragma: no cover
    """Display a message box.

    Parameters
    ----------
    message : str
    title : str
    level : str
        information, warning, or critical

    """
    getattr(QMessageBox, level, 'information')(None, title, message)


class QtDialogLogger(logging.Handler):
    """Display a message box for all errors."""
    def emit(self, record):  # pragma: no cover
        msg = self.format(record)
        message_box(msg, title='An error has occurred', level='critical')


@mockable
def input_dialog(title, sentence, text=None):  # pragma: no cover
    """Display a dialog with a text box.

    Parameters
    ----------

    title : str
        Title of the dialog.
    sentence : str
        Message of the dialog.
    text : str
        Default text in the text box.

    """
    return QInputDialog.getText(None, title, sentence, text=text)


@mockable
def show_box(box):  # pragma: no cover
    """Display a dialog."""
    return _button_name_from_enum(box.exec_())


def set_busy(busy):
    """Set a busy or normal cursor in a Qt application."""
    app = create_app()
    if busy:
        app.setOverrideCursor(Qt.WaitCursor)
    else:
        app.restoreOverrideCursor()
        app.restoreOverrideCursor()


@contextmanager
def busy_cursor(activate=True):
    """Context manager displaying a busy cursor during a long command."""
    if not activate:
        yield
    else:
        set_busy(True)
        yield
        set_busy(False)


def screenshot_default_path(widget, dir=None):
    """Return a default path for the screenshot of a widget."""
    from phylib.utils._misc import phy_config_dir
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    name = 'phy_screenshot_%s_%s.png' % (date, widget.__class__.__name__)
    path = (Path(dir) if dir else phy_config_dir() / 'screenshots') / name
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def screenshot(widget, path=None, dir=None):
    """Save a screenshot of a Qt widget to a PNG file.

    By default, the screenshots are saved in `~/.phy/screenshots/`.

    Parameters
    ----------

    widget : Qt widget
        Any widget to capture (including OpenGL widgets).
    path : str or Path
        Path to the PNG file.

    """
    path = path or screenshot_default_path(widget, dir=dir)
    path = Path(path).resolve()
    if isinstance(widget, QOpenGLWindow):
        # Special call for OpenGL widgets.
        widget.grabFramebuffer().save(str(path))
    else:
        # Generic call for regular Qt widgets.
        widget.grab().save(str(path))
    logger.info("Saved screenshot to %s.", path)
    return path


@require_qt
def screen_size():
    """Return the screen size as a tuple (width, height)."""
    screen = QGuiApplication.primaryScreen()
    geometry = screen.geometry()
    return (geometry.width(), geometry.height())


@require_qt
def is_high_dpi():
    """Return whether the screen has a high density.

    Note: currently, this only returns whether the screen width is greater than an arbitrary
    value chosen at 3000.

    """
    return screen_size()[0] > 3000


# -----------------------------------------------------------------------------
# Widgets
# -----------------------------------------------------------------------------

def _static_abs_path(rel_path):
    """Return the absolute path of a static file saved in this repository."""
    return Path(__file__).parent / 'static' / rel_path


class WebPage(QWebEnginePage):
    """A Qt web page widget."""
    _raise_on_javascript_error = False

    def javaScriptConsoleMessage(self, level, msg, line, source):
        super(WebPage, self).javaScriptConsoleMessage(level, msg, line, source)
        msg = "[JS:L%02d] %s" % (line, msg)
        f = (partial(logger.log, 5), logger.warning, logger.error)[level]
        if self._raise_on_javascript_error and level >= 2:
            raise RuntimeError(msg)
        f(msg)


class WebView(QWebEngineView):
    """A generic HTML widget."""

    def __init__(self, *args):
        super(WebView, self).__init__(*args)
        self.html = None
        assert isinstance(self.window(), QWidget)
        self._page = WebPage(self)
        self.setPage(self._page)
        self.move(100, 100)
        self.resize(400, 400)

    def set_html(self, html, callback=None):
        """Set the HTML code."""
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
# Threading
# -----------------------------------------------------------------------------

class WorkerSignals(QObject):
    """Object holding some signals for the workers."""
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


def thread_pool():
    """Return a QThreadPool instance that can `start()` Worker instances for multithreading.

    Example
    -------

    ```python
    w = Worker(print, "hello world")
    thread_pool().start(w)
    ```

    """
    return QThreadPool.globalInstance()


class Worker(QRunnable):
    """A task (just a Python function) running in the thread pool.

    Constructor
    -----------

    fn : function
    *args : function positional arguments
    **kwargs : function keyword arguments

    """
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):  # pragma: no cover
        """Run the task in a background thread. Should not be called directly unless you want
        to bypass the thread pool."""
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


class Debouncer(object):
    """Debouncer to work in a Qt application.

    Jobs are submitted at given times. They are executed immediately if the
    delay since the last submission is greater than some threshold. Otherwise, execution
    is delayed until the delay since the last submission is greater than the threshold.
    During the waiting time, all submitted jobs erase previous jobs in the queue, so
    only the last jobs are taken into account.

    This is used when multiple row selections are done in an HTML table, and each row
    selection is taking a perceptible time to finish.

    Constructor
    -----------

    delay : int
        The minimal delay between the execution of two successive actions.

    Example
    -------

    ```python
    d = Debouncer(delay=250)
    for i in range(10):
        d.submit(print, "hello world", i)
    d.trigger()  # show "hello world 0" and "hello world 9" after a delay

    ```

    """

    _log_level = 5
    delay = 500

    def __init__(self, delay=None):
        self.delay = delay or self.delay  # minimum delay between job executions, in ms.
        self._last_submission_time = 0
        self.is_waiting = False  # whether we're already waiting for the end of the interactions
        self.pending_functions = {}  # assign keys to pending functions.
        self._timer = QTimer()
        self._timer.timeout.connect(self._timer_callback)

    def _elapsed_enough(self):
        """Return whether the elapsed time since the last submission is greater
        than the threshold."""
        return default_timer() - self._last_submission_time > self.delay * .001

    def _timer_callback(self):
        """Callback for the timer."""
        if self._elapsed_enough():
            logger.log(self._log_level, "Stop waiting and triggering.")
            self._timer.stop()
            self.trigger()

    def submit(self, f, *args, key=None, **kwargs):
        """Submit a function call. Execute immediately if the delay since the last submission
        is higher than the threshold, or wait until executing it otherwiser."""
        self.pending_functions[key] = (f, args, kwargs)
        if self._elapsed_enough():
            logger.log(self._log_level, "Triggering action immediately.")
            # Trigger the action immediately if the delay since the last submission is greater
            # than the threshold.
            self.trigger()
        else:
            logger.log(self._log_level, "Waiting...")
            # Otherwise, we start the timer.
            if not self._timer.isActive():
                self._timer.start(25)
        self._last_submission_time = default_timer()

    def trigger(self):
        """Execute the pending actions."""
        for key, item in self.pending_functions.items():
            if item is None:
                continue
            f, args, kwargs = item
            logger.log(self._log_level, "Trigger %s.", f.__name__)
            f(*args, **kwargs)
            self.pending_functions[key] = None


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
        """Call a function after a delay, unless another function is set in the meantime."""
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
