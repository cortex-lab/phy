# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from vispy import app

from ..qt import wrap_qt
from ..dock import DockWindow
from ...utils._color import _random_color
from ...utils.logging import set_level


# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    set_level('info')


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


@wrap_qt
def test_dock_1():

    gui = DockWindow()

    @gui.shortcut('quit', 'ctrl+q')
    def quit():
        gui.close()

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.show()
    yield

    assert len(gui.list_views('view')) == 2
    gui.close()
    yield


@wrap_qt
def test_dock_state():
    _gs = None
    gui = DockWindow()

    @gui.shortcut('press', 'ctrl+g')
    def press():
        pass

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.add_view(_create_canvas(), 'view2')

    @gui.connect_
    def on_close_gui():
        global _gs
        _gs = gui.save_geometry_state()

    gui.show()
    yield

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
    def on_show():
        print(_gs)
        gui.restore_geometry_state(_gs)

    gui.show()
    yield

    gui.close()
    yield
