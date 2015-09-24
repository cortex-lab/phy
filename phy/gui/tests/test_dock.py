# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark, yield_fixture

from ..qt import Qt
from ..dock import (DockWindow, _show_shortcuts, Actions, Snippets,
                    _parse_snippet)
from phy.utils._color import _random_color
from phy.utils.testing import captured_output

# Skip these tests in "make test-quick".
pytestmark = mark.long


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
    yield DockWindow(position=(200, 100), size=(100, 100))


@yield_fixture
def actions():
    yield Actions()


@yield_fixture
def snippets():
    yield Snippets()


#------------------------------------------------------------------------------
# Test actions
#------------------------------------------------------------------------------

def test_shortcuts():
    shortcuts = {
        'test_1': 'ctrl+t',
        'test_2': ('ctrl+a', 'shift+b'),
    }
    with captured_output() as (stdout, stderr):
        _show_shortcuts(shortcuts, 'test')
    assert 'ctrl+a, shift+b' in stdout.getvalue()


def test_actions(actions):
    actions.add('test', lambda: None)
    # Adding an action twice has no effect.
    actions.add('test', lambda: None)

    # Create a shortcut and display it.
    _captured = []

    @actions.shortcut('h')
    def show_my_shortcuts():
        with captured_output() as (stdout, stderr):
            actions.show_shortcuts()
        _captured.append(stdout.getvalue())

    actions.show_my_shortcuts()
    assert 'show_my_shortcuts' in _captured[0]
    assert ': h' in _captured[0]
    print(_captured[0])

    actions.remove_all()


def test_actions_dock(qtbot, gui, actions):
    actions.attach(gui)
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    _press = []

    @actions.shortcut('ctrl+g')
    def press():
        _press.append(0)

    qtbot.keyPress(gui, Qt.Key_G, Qt.ControlModifier)
    assert _press == [0]


#------------------------------------------------------------------------------
# Test snippets
#------------------------------------------------------------------------------

def test_snippets_parse():
    def _check(args, expected):
        snippet = 'snip ' + args
        assert _parse_snippet(snippet) == ['snip'] + expected

    _check('a', ['a'])
    _check('abc', ['abc'])
    _check('a,b,c', [('a', 'b', 'c')])
    _check('a b,c', ['a', ('b', 'c')])

    _check('1', [1])
    _check('10', [10])

    _check('1.', [1.])
    _check('10.', [10.])
    _check('10.0', [10.0])

    _check('0 1', [0, 1])
    _check('0 1.', [0, 1.])
    _check('0 1.0', [0, 1.])

    _check('0,1', [(0, 1)])
    _check('0,10.', [(0, 10.)])
    _check('0. 1,10.', [0., (1, 10.)])

    _check('2-7', [(2, 3, 4, 5, 6, 7)])
    _check('2 3-5', [2, (3, 4, 5)])

    _check('a b,c d,2 3-5', ['a', ('b', 'c'), ('d', 2), (3, 4, 5)])


def test_snippets(snippets):
    # TODO
    pass


def test_snippets_dock(qtbot, gui, snippets):
    pass


#------------------------------------------------------------------------------
# Test dock
#------------------------------------------------------------------------------

def test_dock_1(qtbot):

    gui = DockWindow(position=(200, 100), size=(100, 100))
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

    # Check that the close_widget event is fired when the dock widget is
    # closed.
    _close = []

    @view.connect_
    def on_close_widget():
        _close.append(0)
    view.close()
    assert _close == [0]

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
