# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from vispy import app

from ..dock import (DockWindow,
                    qt_app,
                    _close_qt_after,
                    )
from .._color import _random_color


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

    @c.connect
    def on_key_press(e):
        c.color = _random_color()
        c.update()

    return c


def test_dock():
    with qt_app():

        gui = DockWindow()

        @gui.shortcut('press', 'ctrl+g')
        def press():
            pass

        gui.add_view(_create_canvas(), 'view1')
        gui.add_view(_create_canvas(), 'view2')
        assert len(gui.list_views('view')) == 2
        _close_qt_after(gui, 0.1)

        gui.show()


def test_dock_state():
    with qt_app():

        gui = DockWindow()

        @gui.shortcut('press', 'ctrl+g')
        def press():
            pass

        gui.add_view(_create_canvas(), 'view1')
        gui.add_view(_create_canvas(), 'view2')
        gui.add_view(_create_canvas(), 'view2')
        assert len(gui.list_views('view')) == 3

        assert gui.view_counts() == {
            'view1': 1,
            'view2': 2,
        }

        _close_qt_after(gui, .1)

        gui.show()
