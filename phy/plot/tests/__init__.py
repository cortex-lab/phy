from phy.gui.qt import Qt, QPoint, _wait


def mouse_click(qtbot, c, pos, button='left', modifiers=()):
    b = getattr(Qt, button.capitalize() + 'Button')
    modifiers = _modifiers_flag(modifiers)
    qtbot.mouseClick(c, b, modifiers, QPoint(*pos))


def mouse_press(qtbot, c, pos, button='left', modifiers=()):
    b = getattr(Qt, button.capitalize() + 'Button')
    modifiers = _modifiers_flag(modifiers)
    qtbot.mousePress(c, b, modifiers, QPoint(*pos))


def mouse_drag(qtbot, c, p0, p1, button='left', modifiers=()):
    b = getattr(Qt, button.capitalize() + 'Button')
    modifiers = _modifiers_flag(modifiers)
    qtbot.mousePress(c, b, modifiers, QPoint(*p0))
    qtbot.mouseMove(c, QPoint(*p1))
    qtbot.mouseRelease(c, b, modifiers, QPoint(*p1))


def _modifiers_flag(modifiers):
    out = Qt.NoModifier
    for m in modifiers:
        out |= getattr(Qt, m + 'Modifier')
    return out


def key_press(qtbot, c, key, modifiers=(), delay=50):
    qtbot.keyPress(c, getattr(Qt, 'Key_' + key), _modifiers_flag(modifiers))
    _wait(delay)


def key_release(qtbot, c, key, modifiers=()):
    qtbot.keyRelease(c, getattr(Qt, 'Key_' + key), _modifiers_flag(modifiers))
