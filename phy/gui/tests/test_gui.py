# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..qt import Qt, QApplication, QWidget, QMessageBox
from ..gui import (GUI, GUIState,
                   _try_get_matplotlib_canvas,
                   _try_get_opengl_canvas,
                   )
from phy.plot import BaseCanvas
from phy.utils import Bunch, connect, unconnect


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

def _create_canvas():
    """Create a GL view."""
    c = BaseCanvas()
    return c


#------------------------------------------------------------------------------
# Test views
#------------------------------------------------------------------------------

def test_matplotlib_view():
    from matplotlib.pyplot import Figure
    assert isinstance(_try_get_matplotlib_canvas(Figure()), QWidget)
    assert isinstance(_try_get_opengl_canvas(Figure()), Figure)

    class MyFigure(object):
        figure = Figure()
    assert isinstance(_try_get_matplotlib_canvas(MyFigure()), QWidget)


def test_opengl_view():
    from phy.plot.base import BaseCanvas
    assert isinstance(_try_get_opengl_canvas(BaseCanvas()), QWidget)
    assert isinstance(_try_get_matplotlib_canvas(BaseCanvas()), BaseCanvas)

    class MyFigure(object):
        canvas = BaseCanvas()
    assert isinstance(_try_get_opengl_canvas(MyFigure()), QWidget)


#------------------------------------------------------------------------------
# Test GUI
#------------------------------------------------------------------------------

def test_gui_noapp(tempdir):
    if not QApplication.instance():
        with raises(RuntimeError):  # pragma: no cover
            GUI(config_dir=tempdir)


def test_gui_1(tempdir, qtbot):

    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir)
    qtbot.addWidget(gui)

    assert gui.name == 'GUI'

    # Increase coverage.
    @connect(sender=gui)
    def on_show():
        pass
    unconnect(on_show)
    qtbot.keyPress(gui, Qt.Key_Control)
    qtbot.keyRelease(gui, Qt.Key_Control)

    assert isinstance(gui.dialog("Hello"), QMessageBox)

    view = gui.add_view(_create_canvas(), floating=True, closable=True)
    gui.add_view(_create_canvas())
    view.setFloating(False)
    gui.show()

    assert gui.view_name(_create_canvas()) == 'BaseCanvas'
    assert gui.get_view(BaseCanvas)
    assert len(gui.list_views(BaseCanvas)) == 2

    # Check that the close_widget event is fired when the gui widget is
    # closed.
    _close = []

    @connect(sender=view)
    def on_close_dock_widget(sender):
        _close.append(0)

    @connect(sender=gui)
    def on_close_view(gui, v):
        _close.append(1)

    view.close()
    assert _close == [1, 0]

    gui.close()

    assert gui.state.geometry_state['geometry']
    assert gui.state.geometry_state['state']

    gui.default_actions.show_all_shortcuts()
    gui.default_actions.exit()


def test_gui_status_message(gui):
    assert gui.status_message == ''
    gui.status_message = ':hello world!'
    assert gui.status_message == ':hello world!'

    gui.lock_status()
    gui.status_message = ''
    assert gui.status_message == ':hello world!'
    gui.unlock_status()
    gui.status_message = ''
    assert gui.status_message == ''


def test_gui_geometry_state(tempdir, qtbot):
    _gs = []
    gui = GUI(size=(800, 600), config_dir=tempdir)
    qtbot.addWidget(gui)

    @connect(sender=gui)
    def on_close(sender):
        _gs.append(gui.save_geometry_state())

    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())

    assert len(gui.list_views(BaseCanvas)) == 3

    gui.close()

    # Recreate the GUI with the saved state.
    gui = GUI(config_dir=tempdir)

    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())

    @connect(sender=gui)
    def on_show(sender):
        gui.restore_geometry_state(_gs[0])

    assert gui.restore_geometry_state(None) is None

    assert len(gui.list_views(BaseCanvas)) == 3

    gui.close()


#------------------------------------------------------------------------------
# Test GUI state
#------------------------------------------------------------------------------

def test_gui_state_view(tempdir):
    view = Bunch(name='MyView0')
    gui = Bunch()
    gui.view_name = lambda view: view.name
    state = GUIState(config_dir=tempdir)
    state.update_view_state(view, dict(hello='world'), gui)
    assert not state.get_view_state(Bunch(name='MyView'), gui)
    assert not state.get_view_state(Bunch(name='MyView (1)'), gui)
    assert state.get_view_state(view, gui) == Bunch(hello='world')
