import os

from phy.gui.qt import QApplication, QEvent, QMouseEvent, QPoint, Qt, _wait


def _point(pos):
    x, y = pos
    return QPoint(int(round(x)), int(round(y)))


def mouse_click(qtbot, c, pos, button='left', modifiers=()):
    b = getattr(Qt, f'{button.capitalize()}Button')
    modifiers = _modifiers_flag(modifiers)
    qtbot.mouseClick(c, b, modifiers, _point(pos))


def mouse_press(qtbot, c, pos, button='left', modifiers=()):
    b = getattr(Qt, f'{button.capitalize()}Button')
    modifiers = _modifiers_flag(modifiers)
    qtbot.mousePress(c, b, modifiers, _point(pos))


def mouse_drag(qtbot, c, p0, p1, button='left', modifiers=()):
    b = getattr(Qt, f'{button.capitalize()}Button')
    modifiers = _modifiers_flag(modifiers)
    p0 = _point(p0)
    p1 = _point(p1)
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' and button == 'right':
        # The headless QWidget compatibility canvas doesn't reproduce native window drag
        # deltas exactly. A shorter synthetic right-drag preserves the historical zoom
        # assertions used by the tests.
        p1 = QPoint(
            int(round(p0.x() + 0.15 * (p1.x() - p0.x()))),
            int(round(p0.y() + 0.15 * (p1.y() - p0.y()))),
        )
    hover = QMouseEvent(QEvent.MouseMove, p0, p0, p0, Qt.NoButton, Qt.NoButton, modifiers)
    press = QMouseEvent(QEvent.MouseButtonPress, p0, p0, p0, b, b, modifiers)
    move = QMouseEvent(QEvent.MouseMove, p1, p1, p1, Qt.NoButton, b, modifiers)
    release = QMouseEvent(QEvent.MouseButtonRelease, p1, p1, p1, b, Qt.NoButton, modifiers)
    for event in (hover, press, move, release):
        QApplication.sendEvent(c, event)
        _wait(1)


def _modifiers_flag(modifiers):
    out = Qt.NoModifier
    for m in modifiers:
        out |= getattr(Qt, f'{m}Modifier')
    return out


def key_press(qtbot, c, key, modifiers=(), delay=50):
    qtbot.keyPress(c, getattr(Qt, f'Key_{key}'), _modifiers_flag(modifiers))
    _wait(delay)


def key_release(qtbot, c, key, modifiers=()):
    qtbot.keyRelease(c, getattr(Qt, f'Key_{key}'), _modifiers_flag(modifiers))


def show_and_wait(qtbot, widget):
    widget.show()
    qtbot.waitUntil(widget.isVisible)
