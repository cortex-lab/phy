# -*- coding: utf-8 -*-

"""Test Qt utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from phylib.utils.testing import captured_logging
from ..qt import (
    QMessageBox, Qt, QWebEngineView, QTimer, _button_name_from_enum, _button_enum_from_name,
    _prompt, _screen_size, _is_high_dpi, _wait_signal, require_qt, create_app, QApplication,
    WebView, busy_cursor, AsyncCaller, QThreadPool, _wait, Worker, _block, _screenshot,
    Debouncer)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_require_qt_with_app():

    @require_qt
    def f():
        pass

    if not QApplication.instance():
        with raises(RuntimeError):  # pragma: no cover
            f()


def test_require_qt_without_app(qapp):

    @require_qt
    def f():
        pass

    # This should not raise an error.
    f()


def test_qt_app(qtbot):
    create_app()
    view = QWebEngineView()
    qtbot.addWidget(view)
    view.close()


def test_screen_size(qtbot):
    _screen_size()
    assert _is_high_dpi() in (False, True)


def test_worker(qtbot):
    pool = QThreadPool.globalInstance()
    _l = []

    def f():  # pragma: no cover
        _l.append(0)
    w = Worker(f)
    pool.start(w)
    _wait(10)
    assert _l == [0]


def test_debouncer(qtbot):
    d = Debouncer(delay=50)
    _l = []

    def f(i):
        _l.append(i)

    for i in range(10):
        qtbot.wait(10)
        d.submit(f, i)
    qtbot.wait(500)
    assert _l == [0, 9]


def test_block(qtbot):
    create_app()
    with raises(RuntimeError):
        _block(lambda: False, timeout=.1)


def test_wait_signal(qtbot):
    x = []

    def f():
        x.append(0)

    timer = QTimer()
    timer.setInterval(100)
    timer.setSingleShot(True)
    timer.timeout.connect(f)
    timer.start()

    assert x == []

    with _wait_signal(timer.timeout):
        pass
    assert x == [0]


def test_web_view(qtbot):

    view = WebView()

    def _assert(text):
        return view.html == '<html><head></head><body>%s</body></html>' % text

    view.set_html('hello', _assert)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitForWindowShown(view)
    _block(lambda: _assert('hello'))

    view.set_html("world")
    _block(lambda: _assert('world'))
    view.close()


def test_javascript_1(qtbot):
    view = WebView()
    with captured_logging() as buf:
        view.set_html('<script>console.log("Test.");</script>')
        qtbot.addWidget(view)
        view.show()
        qtbot.waitForWindowShown(view)
        _block(lambda: view.html is not None)
        view.close()
    assert buf.getvalue() == ""


def test_javascript_2(qtbot):
    view = WebView()
    view._page._raise_on_javascript_error = True
    with qtbot.capture_exceptions() as exceptions:
        view.set_html('<script>console.error("Test.");</script>')
        qtbot.addWidget(view)
        view.show()
        qtbot.waitForWindowShown(view)
        _block(lambda: view.html is not None)
        view.close()
    assert len(exceptions) >= 1


def test_screenshot(qtbot, tempdir):
    path = tempdir / 'capture.png'
    view = WebView()
    view.set_html('hello', lambda e: _screenshot(view, path))
    qtbot.addWidget(view)
    view.show()
    qtbot.waitForWindowShown(view)
    _block(lambda: path.exists())
    view.close()


def test_prompt(qtbot):

    assert _button_name_from_enum(QMessageBox.Save) == 'save'
    assert _button_enum_from_name('save') == QMessageBox.Save

    box = _prompt("How are you doing?",
                  buttons=['save', 'cancel', 'close'],
                  )
    qtbot.mouseClick(box.buttons()[0], Qt.LeftButton)
    assert 'save' in str(box.clickedButton().text()).lower()


def test_busy_cursor(qtbot):
    with busy_cursor():
        pass


def test_async_caller(qtbot):
    ac = AsyncCaller(delay=10)

    _l = []

    @ac.set
    def f():
        _l.append(0)

    assert not _l

    qtbot.wait(20)

    assert _l == [0]

    @ac.set
    def g():
        _l.append(0)

    qtbot.wait(20)

    assert _l == [0, 0]
