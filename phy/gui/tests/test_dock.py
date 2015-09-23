# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from vispy import app

from ..qt import Qt
from ..dock import DockWindow
from phy.utils._color import _random_color


# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _create_canvas():
    """Create a VisPy canvas with a color background."""
    c = app.Canvas()
    c.color = _random_color()

    @c.connect
    def on_draw(e):
        c.context.clear(c.color)

    return c


def test_dock_1(qtbot):

    gui = DockWindow()
    qtbot.addWidget(gui)

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.show()
    qtbot.waitForWindowShown(gui)

    assert len(gui.list_views('view')) == 2
    gui.close()


def test_dock_status_message(qtbot):
    gui = DockWindow()
    qtbot.addWidget(gui)
    assert gui.status_message == ''
    gui.status_message = ':hello world!'
    assert gui.status_message == ':hello world!'


def test_dock_state(qtbot):
    _gs = []
    gui = DockWindow(size=(100, 100))
    qtbot.addWidget(gui)

    _press = []

    @gui.shortcut('press', 'ctrl+g')
    def press():
        _press.append(0)

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.add_view(_create_canvas(), 'view2')

    @gui.connect_
    def on_close_gui():
        _gs.append(gui.save_geometry_state())

    gui.show()
    qtbot.waitForWindowShown(gui)

    qtbot.keyPress(gui, Qt.Key_G, Qt.ControlModifier)
    assert _press == [0]

    assert len(gui.list_views('view')) == 3
    assert gui.view_count() == {
        'view1': 1,
        'view2': 2,
    }

    gui.close()

    # Recreate the GUI with the saved state.
    gui = DockWindow()

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.add_view(_create_canvas(), 'view2')

    @gui.connect_
    def on_show_gui():
        gui.restore_geometry_state(_gs[0])

    qtbot.addWidget(gui)
    gui.show()

    assert len(gui.list_views('view')) == 3
    assert gui.view_count() == {
        'view1': 1,
        'view2': 2,
    }

    gui.close()
