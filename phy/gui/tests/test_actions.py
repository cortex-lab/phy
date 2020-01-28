# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial

from pytest import raises

from ..actions import (
    _show_shortcuts, _show_snippets, _get_shortcut_string, _get_qkeysequence, _parse_snippet,
    _expected_args, Actions)
from phylib.utils.testing import captured_output, captured_logging
from ..qt import mock_dialogs


#------------------------------------------------------------------------------
# Test actions
#------------------------------------------------------------------------------

def test_expected_args():
    assert _expected_args(lambda: 0) == ()
    assert _expected_args(lambda a: 0) == ('a',)
    assert _expected_args(lambda a, b: 0) == ('a', 'b')
    assert _expected_args(lambda self, a: 0) == ('a',)
    assert _expected_args(lambda a, b=0: 0) == ('a',)
    assert _expected_args(partial(lambda a, b: 0, 0)) == ('b',)


def test_shortcuts(qapp):
    assert 'z' in _get_shortcut_string('Undo')

    def _assert_shortcut(name, key=None):
        shortcut = _get_qkeysequence(name)
        s = _get_shortcut_string(shortcut)
        if key is None:
            assert s == s
        else:
            assert key in s

    _assert_shortcut('Undo', 'z')
    _assert_shortcut('Save', 's')
    _assert_shortcut('q')
    _assert_shortcut('ctrl+q')
    _assert_shortcut(':')
    _assert_shortcut(['ctrl+a', 'shift+b'])


def test_show_shortcuts(qapp):
    # NOTE: a Qt application needs to be running so that we can use the
    # KeySequence.
    shortcuts = {
        'test_1': 'ctrl+t',
        'test_2': ('ctrl+a', 'shift+b'),
        'test_3': 'ctrl+z',
    }
    with captured_output() as (stdout, stderr):
        _show_shortcuts(shortcuts)
    assert 'ctrl+a, shift+b' in stdout.getvalue()
    assert 'ctrl+z' in stdout.getvalue()


def test_show_snippets():
    snippets = {
        'test_1 (note)': 't1',
    }
    with captured_output() as (stdout, stderr):
        _show_snippets(snippets)
    assert ':t1' in stdout.getvalue()


def test_actions_default_shortcuts(gui):
    actions = Actions(gui, default_shortcuts={'my_action': 'a'}, name='actions')
    actions.add(lambda: None, name='my_action')
    assert actions.shortcuts['my_action'] == 'a'


def test_actions_simple(actions):

    _res = []

    def _action(*args):
        _res.append(args)

    actions.add(_action, 'tes&t')
    # Adding an action twice has no effect.
    actions.add(_action, 'test')

    # Create a shortcut and display it.
    _captured = []

    @actions.add(shortcut='h')
    def show_my_shortcuts():
        with captured_output() as (stdout, stderr):
            actions.show_shortcuts()
        _captured.append(stdout.getvalue())

    actions.show_my_shortcuts()
    assert 'show_my_shortcuts' in _captured[0]
    assert 'h\n' in _captured[0]

    actions.run('t', 1)
    assert _res == [(1,)]

    assert 'show_my_shortcuts' in actions
    assert 'unknown' not in actions

    actions.remove_all()

    assert '<Actions' in actions.__repr__()


#------------------------------------------------------------------------------
# Test actions and snippet
#------------------------------------------------------------------------------

def test_actions_gui_menu(qtbot, gui, actions):
    qtbot.addWidget(gui)

    @actions.add(shortcut='g', menu='&File', icon='f0c7', toolbar=True)
    def press():
        print("press")

    actions.separator(menu='&File')

    gui.show()
    qtbot.waitForWindowShown(gui)

    actions.get('press').trigger()

    gui.close()
    # qtbot.stop()


def test_actions_gui(qtbot, gui, actions):
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    _press = []

    @actions.add(shortcut='g')
    def press():
        _press.append(0)

    actions.press()
    assert _press == [0]

    # Show action shortcuts.
    with captured_output() as (stdout, stderr):
        actions.show_shortcuts()
    assert 'g\n' in stdout.getvalue()

    # Show default action shortcuts.
    with captured_output() as (stdout, stderr):
        gui.file_actions.show_shortcuts()
    assert 'q\n' in stdout.getvalue()

    # Show all action shortcuts.
    with captured_output() as (stdout, stderr):
        gui.help_actions.show_all_shortcuts()
    assert 'g\n' in stdout.getvalue()


def test_actions_submenu(qtbot, gui, actions):
    @actions.add(menu='&File', submenu='Submenu')
    def my_action():
        pass

    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    # qtbot.stop()
    gui.close()


def test_actions_checkable(qtbot, gui, actions):
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    _l = []

    @actions.add(shortcut='c', checkable=True, menu='&File')
    def toggle(checked):
        _l.append(checked)

    actions.get('toggle').trigger()
    actions.get('toggle').trigger()
    assert _l == [True, False]


def test_actions_dialog_1(qtbot, gui, actions):

    @actions.add(shortcut='a', prompt=True)
    def hello(arg):
        print("hello", arg)

    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    with captured_output() as (stdout, stderr):
        with mock_dialogs(('world', True)):
            # return string, ok
            actions.get('hello').trigger()
    assert 'hello world' in stdout.getvalue()


def test_actions_dialog_2(qtbot, gui, actions):

    @actions.add(shortcut='a', prompt=True)
    def hello(arg1, arg2):
        print("hello", arg1, arg2)

    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    with captured_output() as (stdout, stderr):
        with mock_dialogs(('world world', True)):
            # return string, ok
            actions.get('hello').trigger()
    assert 'hello world' in stdout.getvalue()

    with captured_logging('phy.gui.actions') as buf:
        with mock_dialogs(('world', True)):
            actions.get('hello').trigger()
    assert 'invalid' in buf.getvalue().lower()


def test_actions_disable(qtbot, gui, actions):
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    @actions.add(shortcut='g')
    def press():
        pass

    actions.disable()
    assert not actions.get('press').isEnabled()

    actions.enable()
    assert actions.get('press').isEnabled()


def test_snippets_message(qtbot, gui):
    gui.status_message = 'Hello world!'
    gui.snippets.mode_on()
    gui.snippets.mode_off()
    assert gui.status_message == 'Hello world!'


def test_snippets_gui(qtbot, gui, actions):
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    _actions = []

    @actions.add(name='my_test_1', alias='t1')
    def test(*args):
        _actions.append(args)

    # Attach the GUI and register the actions.
    snippets = gui.snippets

    # Simulate the following keystrokes `:t2 ^H^H1 3-5 ab,c `
    assert not snippets.is_mode_on()

    def _run(cmd):
        """Simulate keystrokes."""
        for char in cmd:
            i = snippets._snippet_chars.index(char)
            snippets.actions.run('_snippet_{}'.format(i))

    snippets.actions.enable_snippet_mode()
    _run('t2 ')
    assert snippets.is_mode_on()
    snippets.actions._snippet_backspace()
    snippets.actions._snippet_backspace()
    _run('1 3-5 ab,c')
    snippets.actions._snippet_activate()

    assert _actions == [([3, 4, 5], ['ab', 'c'])]


#------------------------------------------------------------------------------
# Test snippets
#------------------------------------------------------------------------------

def test_snippets_parse():
    def _check(args, expected):
        snippet = 'snip ' + args
        assert _parse_snippet(snippet) == tuple(['snip'] + expected)

    _check('a', ['a'])
    _check('abc', ['abc'])
    _check('a,b,c', [['a', 'b', 'c']])
    _check('a b,C', ['a', ['b', 'C']])

    _check('1', [1])
    _check('10', [10])

    _check('1.', [1.])
    _check('10.', [10.])
    _check('10.0', [10.0])

    _check('0 1', [0, 1])
    _check('0 1.', [0, 1.])
    _check('0 1.0', [0, 1.])

    _check('0,1', [[0, 1]])
    _check('0,10.', [[0, 10.]])
    _check('0. 1,10.', [0., [1, 10.]])

    _check('2-7', [[2, 3, 4, 5, 6, 7]])
    _check('2 3-5', [2, [3, 4, 5]])

    _check('a b,c d,2 3-5', ['a', ['b', 'c'], ['d', 2], [3, 4, 5]])


def test_snippets_errors(actions, snippets):

    _actions = []

    @actions.add(name='my_test', alias='t')
    def test(arg):
        # Enforce single-character argument.
        assert len(str(arg)) == 1
        _actions.append(arg)

    logging_name = 'phy.gui.actions'
    with captured_logging(logging_name) as buf:
        snippets.run(':t1')
    assert 'couldn\'t' in buf.getvalue().lower()

    with captured_logging(logging_name) as buf:
        snippets.run(':t')
    assert 'invalid' in buf.getvalue().lower()

    with captured_logging(logging_name) as buf:
        snippets.run(':t 1 2')
    assert 'invalid' in buf.getvalue().lower()

    with captured_logging(logging_name) as buf:
        snippets.run(':t aa')
    assert 'assert 2 == 1' in buf.getvalue()

    snippets.run(':t a')
    assert _actions == ['a']


def test_snippets_actions_1(actions, snippets):

    _actions = []

    @actions.add(name='my_test_1')
    def test_1(*args):
        _actions.append((1, args))

    @actions.add(name='my_&test_2')
    def test_2(*args):
        _actions.append((2, args))

    @actions.add(name='my_test_3', alias='t3')
    def test_3(*args):
        _actions.append((3, args))

    assert snippets.command == ''

    # Action 1.
    snippets.run(':my_test_1')
    assert _actions == [(1, ())]

    # Action 2.
    snippets.run(':t 1.5 a 2-4 5,7')
    assert _actions[-1] == (2, (1.5, 'a', [2, 3, 4], [5, 7]))

    def _run(cmd):
        """Simulate keystrokes."""
        for char in cmd:
            i = snippets._snippet_chars.index(char)
            snippets.actions.run('_snippet_{}'.format(i))

    # Need to activate the snippet mode first.
    with raises(ValueError):
        _run(':t3 hello')

    # Simulate keystrokes ':t3 hello<Enter>'
    snippets.mode_on()  # ':'
    snippets.actions._snippet_backspace()
    _run('t3 hello')
    snippets.actions._snippet_activate()  # 'Enter'
    assert _actions[-1] == (3, ('hello',))
    snippets.mode_off()


def test_snippets_actions_2(actions, snippets):

    _actions = []

    @actions.add
    def test(arg):
        _actions.append(arg)

    actions.test(1)
    assert _actions == [1]

    snippets.mode_on()
    snippets.mode_off()

    actions.test(2)
    assert _actions == [1, 2]
