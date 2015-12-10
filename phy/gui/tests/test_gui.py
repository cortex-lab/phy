# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..qt import Qt, QApplication, QWidget
from ..gui import (GUI, load_gui_plugins,
                   _try_get_matplotlib_canvas,
                   _try_get_vispy_canvas,
                   )
from phy.utils import IPlugin
from phy.utils._color import _random_color


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

def _create_canvas():
    """Create a VisPy canvas with a color background."""
    from vispy import app
    c = app.Canvas()
    c.color = _random_color()

    @c.connect
    def on_draw(e):  # pragma: no cover
        c.context.clear(c.color)

    return c


#------------------------------------------------------------------------------
# Test views
#------------------------------------------------------------------------------

def test_vispy_view():
    from vispy.app import Canvas
    assert isinstance(_try_get_vispy_canvas(Canvas()), QWidget)


def test_matplotlib_view():
    from matplotlib.pyplot import Figure
    assert isinstance(_try_get_matplotlib_canvas(Figure()), QWidget)


#------------------------------------------------------------------------------
# Test GUI
#------------------------------------------------------------------------------

def test_gui_noapp():
    if not QApplication.instance():
        with raises(RuntimeError):  # pragma: no cover
            GUI()


def test_gui_1(qtbot):

    gui = GUI(position=(200, 100), size=(100, 100))
    qtbot.addWidget(gui)

    assert gui.name == 'GUI'

    # Increase coverage.
    @gui.connect_
    def on_show():
        pass
    gui.unconnect_(on_show)
    qtbot.keyPress(gui, Qt.Key_Control)
    qtbot.keyRelease(gui, Qt.Key_Control)

    view = gui.add_view(_create_canvas(), 'view1', floating=True)
    gui.add_view(_create_canvas(), 'view2')
    view.setFloating(False)
    gui.show()

    assert len(gui.list_views('view')) == 2

    # Check that the close_widget event is fired when the gui widget is
    # closed.
    _close = []

    @view.connect_
    def on_close_widget():
        _close.append(0)
    view.close()
    assert _close == [0]

    gui.default_actions.exit()


def test_load_gui_plugins(gui, tempdir):

    load_gui_plugins(gui)

    _tmp = []

    class MyPlugin(IPlugin):
        def attach_to_gui(self, gui, session):
            _tmp.append(session)

    load_gui_plugins(gui, plugins=['MyPlugin'], session='hello')

    assert _tmp == ['hello']


def test_gui_component(gui):

    class TestComponent(object):
        def __init__(self, arg):
            self._arg = arg

        def attach(self, gui):
            gui._attached = self._arg
            return 'attached'

    tc = TestComponent(3)

    assert tc.attach(gui) == 'attached'
    assert gui._attached == 3


def test_gui_status_message(gui):
    assert gui.status_message == ''
    gui.status_message = ':hello world!'
    assert gui.status_message == ':hello world!'


def test_gui_state(qtbot):
    _gs = []
    gui = GUI(size=(100, 100))
    qtbot.addWidget(gui)

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.add_view(_create_canvas(), 'view2')

    @gui.connect_
    def on_close():
        _gs.append(gui.save_geometry_state())

    gui.show()
    qtbot.waitForWindowShown(gui)

    assert len(gui.list_views('view')) == 3
    assert gui.view_count() == {
        'view1': 1,
        'view2': 2,
    }

    gui.close()

    # Recreate the GUI with the saved state.
    gui = GUI()

    gui.add_view(_create_canvas(), 'view1')
    gui.add_view(_create_canvas(), 'view2')
    gui.add_view(_create_canvas(), 'view2')

    @gui.connect_
    def on_show():
        gui.restore_geometry_state(_gs[0])

    assert gui.restore_geometry_state(None) is None

    qtbot.addWidget(gui)
    gui.show()

    assert len(gui.list_views('view')) == 3
    assert gui.view_count() == {
        'view1': 1,
        'view2': 2,
    }

    gui.close()
