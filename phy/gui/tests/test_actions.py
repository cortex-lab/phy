# -*- coding: utf-8 -*-

"""Test dock."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark, raises, yield_fixture

from ..actions import _show_shortcuts, Actions, Snippets, _parse_snippet
from phy.utils._color import _random_color
from phy.utils.testing import captured_output, captured_logging

# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

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


def test_actions_simple(actions):

    _res = []

    def _action(*args):
        _res.append(args)

    actions.add('tes&t', _action)
    # Adding an action twice has no effect.
    actions.add('test', _action)

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

    with raises(ValueError):
        assert actions.get_name('e')
    assert actions.get_name('t') == 'test'
    assert actions.get_name('test') == 'test'

    actions.run('t', 1)
    assert _res == [(1,)]

    # Run an action instance.
    actions.run(actions._actions['test'], 1)

    actions.remove_all()


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


def test_snippets_errors(actions, snippets):

    _actions = []

    @actions.connect
    def on_reset():
        @actions.shortcut(name='my_test', alias='t')
        def test(arg):
            # Enforce single-character argument.
            assert len(str(arg)) == 1
            _actions.append(arg)

    # Attach the GUI and register the actions.
    snippets.attach(None, actions)
    actions.reset()

    with raises(ValueError):
        snippets.run(':t1')

    with captured_logging() as buf:
        snippets.run(':t')
    assert 'missing 1 required positional argument' in buf.getvalue()

    with captured_logging() as buf:
        snippets.run(':t 1 2')
    assert 'takes 1 positional argument but 2 were given' in buf.getvalue()

    with captured_logging() as buf:
        snippets.run(':t aa')
    assert 'assert 2 == 1' in buf.getvalue()

    snippets.run(':t a')
    assert _actions == ['a']


def test_snippets_actions(actions, snippets):

    _actions = []

    @actions.connect
    def on_reset():
        @actions.shortcut(name='my_test_1')
        def test_1(*args):
            _actions.append((1, args))

        @actions.shortcut(name='my_&test_2')
        def test_2(*args):
            _actions.append((2, args))

        @actions.shortcut(name='my_test_3', alias='t3')
        def test_3(*args):
            _actions.append((3, args))

    # Attach the GUI and register the actions.
    snippets.attach(None, actions)
    actions.reset()

    assert snippets.command == ''

    # Action 1.
    snippets.run(':my_test_1')
    assert _actions == [(1, ())]

    # Action 2.
    snippets.run(':t 1.5 a 2-4 5,7')
    assert _actions[-1] == (2, (1.5, 'a', (2, 3, 4), (5, 7)))

    def _run(cmd):
        """Simulate keystrokes."""
        for char in cmd:
            i = snippets._snippet_chars.index(char)
            actions.run('_snippet_{}'.format(i))

    # Need to activate the snippet mode first.
    with raises(ValueError):
        _run(':t3 hello')

    # Simulate keystrokes ':t3 hello<Enter>'
    snippets.mode_on()  # ':'
    actions._snippet_backspace()
    _run('t3 hello')
    actions._snippet_activate()  # 'Enter'
    assert _actions[-1] == (3, ('hello',))
    snippets.mode_off()
