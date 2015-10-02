# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
from sys import platform

from pytest import mark, yield_fixture

from ..qt import Qt
from ..gui import GUI
from phy.utils._color import _random_color
from phy.utils.plugin import IPlugin
from .test_actions import actions, snippets  # noqa

# Skip some tests on OS X or on CI systems (Travis).
skip_mac = mark.skipif(platform == "darwin",
                       reason="Some tests don't work on OS X because of a bug "
                              "with QTest (qtbot) keyboard events that don't "
                              "trigger QAction shortcuts. On CI these tests "
                              "fail because the GUI is not displayed.")

skip_ci = mark.skipif(os.environ.get('CI', None) is not None,
                      reason="Some shortcut-related Qt tests fail on CI.")


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


@yield_fixture
def gui():
    yield GUI(position=(200, 100), size=(100, 100))


#------------------------------------------------------------------------------
# Test actions and snippet
#------------------------------------------------------------------------------

@skip_mac  # noqa
@skip_ci
def test_actions_gui(qtbot, gui, actions):
    actions.attach(gui)

    # Set the default actions.
    actions.reset()

    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    _press = []

    @actions.shortcut('ctrl+g')
    def press():
        _press.append(0)

    qtbot.keyPress(gui, Qt.Key_G, Qt.ControlModifier)
    assert _press == [0]

    # Quit the GUI.
    qtbot.keyPress(gui, Qt.Key_Q, Qt.ControlModifier)


@skip_mac  # noqa
@skip_ci
def test_snippets_gui(qtbot, gui, actions, snippets):

    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    _actions = []

    @actions.connect
    def on_reset():
        @actions.shortcut(name='my_test_1', alias='t1')
        def test(*args):
            _actions.append(args)

    # Attach the GUI and register the actions.
    snippets.attach(gui, actions)
    actions.attach(gui)
    actions.reset()

    # Simulate the following keystrokes `:t2 ^H^H1 3-5 ab,c `
    assert not snippets.is_mode_on()
    qtbot.keyClicks(gui, ':t2 ')
    assert snippets.is_mode_on()
    qtbot.keyPress(gui, Qt.Key_Backspace)
    qtbot.keyPress(gui, Qt.Key_Backspace)
    qtbot.keyClicks(gui, '1 3-5 ab,c')
    qtbot.keyPress(gui, Qt.Key_Return)

    assert _actions == [((3, 4, 5), ('ab', 'c'))]


#------------------------------------------------------------------------------
# Test gui
#------------------------------------------------------------------------------

def test_gui_1(qtbot):

    gui = GUI(position=(200, 100), size=(100, 100))
    qtbot.addWidget(gui)

    # Increase coverage.
    @gui.connect_
    def on_show_gui():
        pass
    gui.unconnect_(on_show_gui)
    qtbot.keyPress(gui, Qt.Key_Control)
    qtbot.keyRelease(gui, Qt.Key_Control)

    view = gui.add_view(_create_canvas(), 'view1', floating=True)
    gui.add_view(_create_canvas(), 'view2')
    view.setFloating(False)
    gui.show()
    # qtbot.waitForWindowShown(gui)

    assert len(gui.list_views('view')) == 2

    # Check that the close_widget event is fired when the gui widget is
    # closed.
    _close = []

    @view.connect_
    def on_close_widget():
        _close.append(0)
    view.close()
    assert _close == [0]

    gui.close()


def test_gui_plugin(qtbot, gui):

    class TestPlugin(IPlugin):
        def attach_to_gui(self, gui):
            gui._attached = True
            return 'attached'

    assert gui.attach('testplugin') == 'attached'
    assert gui._attached


def test_gui_status_message(qtbot):
    gui = GUI()
    qtbot.addWidget(gui)
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
    def on_close_gui():
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
