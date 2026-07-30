"""Test Qt utilities."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

from pytest import raises

from ..qt import (
    AsyncCaller,
    Debouncer,
    QApplication,
    QMessageBox,
    Qt,
    QTimer,
    QWidget,
    Worker,
    _block,
    _button_enum_from_name,
    _button_name_from_enum,
    _wait,
    _wait_signal,
    busy_cursor,
    create_app,
    is_high_dpi,
    prompt,
    require_qt,
    screen_size,
    screenshot,
    screenshot_default_path,
    thread_pool,
)
from . import show_and_wait

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------


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
    view = QWidget()
    qtbot.addWidget(view)
    view.close()


def test_screen_size(qtbot):
    screen_size()
    assert is_high_dpi() in (False, True)


def test_worker(qtbot):
    pool = thread_pool()
    _l = []

    def f():  # pragma: no cover
        _l.append(0)

    w = Worker(f)
    pool.start(w)
    _wait(10)
    assert _l == [0]


def test_debouncer_1(qtbot):
    d = Debouncer(delay=50)
    _l = []

    def f(i):
        _l.append(i)

    for i in range(10):
        qtbot.wait(10)
        d.submit(f, i)
    qtbot.wait(500)
    assert _l == [0, 9]


def test_debouncer_2(qtbot):
    d = Debouncer(delay=100)
    _l = []

    def f(i):
        _l.append(i)

    # Step 1: without stop_waiting.
    d.submit(f, 0)
    d.submit(f, 1)
    d.submit(f, 2)
    qtbot.wait(50)

    assert _l == [0]

    # Reset
    qtbot.waitUntil(lambda: _l == [0, 2], timeout=500)
    _l.clear()

    # Step 2: with stop_waiting.
    d.submit(f, 0)
    d.submit(f, 1)
    d.submit(f, 2)
    assert _l == [0]

    d.stop_waiting(0.001)
    qtbot.waitUntil(lambda: _l == [0, 2], timeout=500)


def test_debouncer_flush(qtbot):
    d = Debouncer(delay=1000)
    calls = []

    d.submit(calls.append, ('first', 0), key='first')
    d.submit(calls.append, ('first', 1), key='first')
    d.submit(calls.append, ('second', 0), key='second')

    assert calls == [('first', 0)]
    assert d.has_pending
    assert d._timer.isActive()

    d.flush()

    assert calls == [('first', 0), ('first', 1), ('second', 0)]
    assert not d.has_pending
    assert not d._timer.isActive()

    # Flushing again must not repeat an action.
    d.flush()
    assert calls == [('first', 0), ('first', 1), ('second', 0)]


def test_debouncer_zero_delay():
    d = Debouncer(delay=0)
    calls = []

    d.submit(calls.append, 0)
    d.submit(calls.append, 1)

    assert d.delay == 0
    assert calls == [0, 1]


def test_debouncer_flush_preserves_reentrant_submission(qtbot):
    d = Debouncer(delay=1000)
    calls = []

    def pending():
        calls.append('pending')
        d.submit(calls.append, 'reentrant', key='action')

    d.submit(calls.append, 'first')
    d.submit(pending, key='action')
    assert calls == ['first']

    d.flush()
    assert calls == ['first', 'pending']
    assert d.has_pending

    d.flush()
    assert calls == ['first', 'pending', 'reentrant']
    assert not d.has_pending


def test_block(qtbot):
    create_app()
    with raises(RuntimeError):
        _block(lambda: False, timeout=0.1)


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


def test_screenshot(qtbot, tempdir):
    path = tempdir / 'capture.png'
    view = QWidget()
    assert str(screenshot_default_path(view, dir=tempdir)).startswith(str(tempdir))
    qtbot.addWidget(view)
    show_and_wait(qtbot, view)
    screenshot(view, path)
    _block(lambda: path.exists())
    view.close()


def test_prompt(qtbot):
    assert _button_name_from_enum(QMessageBox.Save) == 'save'
    assert _button_enum_from_name('save') == QMessageBox.Save

    box = prompt('How are you doing?', buttons=['save', 'cancel', 'close'])
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
