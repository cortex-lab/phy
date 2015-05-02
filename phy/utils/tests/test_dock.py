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

        @gui.shortcut('press', 'ctrl+g')
        def press():
            pass

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

    # Recreate the GUI with the saved state.
    with qt_app():
        gui = DockWindow()

        gui.add_view(_create_canvas(), 'view1')
        gui.add_view(_create_canvas(), 'view2')
        gui.add_view(_create_canvas(), 'view2')

        @gui.on_show
        def on_show():
            gui.restore_geometry_state(gs)

        _show(gui)
