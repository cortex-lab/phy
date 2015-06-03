# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from vispy import app

from ..qt import (qt_app,
                  _close_qt_after,
                  _prompt,
                  )
from ..dock import DockWindow
from ...utils._color import _random_color


# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

_DURATION = .1


def _create_canvas():
    """Create a VisPy canvas with a color background."""
    c = app.Canvas()
    c.color = _random_color()

    @c.connect
    def on_draw(e):
        c.context.clear(c.color)

    @c.connect
    def on_key_press(e):
        c.color = _random_color()
        c.update()

    return c


def _show(gui):
    _close_qt_after(gui, _DURATION)
    gui.show()
    return gui


def test_dock_1():
    with qt_app():

        gui = DockWindow()

        @gui.shortcut('quit', 'ctrl+q')
        def quit():
            gui.close()

        gui.add_view(_create_canvas(), 'view1')
        gui.add_view(_create_canvas(), 'view2')
        _show(gui)
        assert len(gui.list_views('view')) == 2


def test_dock_state():
    with qt_app():

        gui = DockWindow()

        @gui.shortcut('press', 'ctrl+g')
        def press():
            pass

        gui.add_view(_create_canvas(), 'view1')
        gui.add_view(_create_canvas(), 'view2')
        gui.add_view(_create_canvas(), 'view2')

        @gui.on_close
        def on_close():
            gui.gs = gui.save_geometry_state()

        _show(gui)

        assert len(gui.list_views('view')) == 3
        assert gui.view_counts() == {
            'view1': 1,
            'view2': 2,
        }

    gs = gui.gs
    with qt_app():

        # Recreate the GUI with the saved state.
        gui = DockWindow()

        gui.add_view(_create_canvas(), 'view1')
        gui.add_view(_create_canvas(), 'view2')
        gui.add_view(_create_canvas(), 'view2')

        @gui.on_show
        def on_show():
            gui.restore_geometry_state(gs)

        _show(gui)


@mark.skipif
def test_prompt():
    with qt_app():
        gui = DockWindow()
        result = _prompt(gui,
                         "How are you doing?",
                         buttons=['save', 'cancel', 'close'],
                         )
        print(result)
        _show(gui)
