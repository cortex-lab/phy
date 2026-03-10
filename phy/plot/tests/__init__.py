from phy.gui.qt import QPoint, Qt, _wait


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
    qtbot.mousePress(c, b, modifiers, _point(p0))
    qtbot.mouseMove(c, _point(p1))
    qtbot.mouseRelease(c, b, modifiers, _point(p1))


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
