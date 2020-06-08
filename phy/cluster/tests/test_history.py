# -*- coding: utf-8 -*-

"""Tests of history structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .._history import History, GlobalHistory


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_history():
    history = History()
    assert history.current_item is None

    def _assert_current(item):
        assert id(history.current_item) == id(item)

    item0 = np.zeros(3)
    item1 = np.ones(4)
    item2 = 2 * np.ones(5)

    assert not history.is_first()
    assert history.is_last()

    history.add(item0)
    _assert_current(item0)

    history.add(item1)
    _assert_current(item1)

    assert history.back() is not None
    _assert_current(item0)

    assert history.forward() is not None
    _assert_current(item1)

    assert history.forward() is None
    _assert_current(item1)

    assert history.back() is not None
    _assert_current(item0)
    assert history.back() is not None
    assert history.back() is None
    assert len(history) == 3

    history.add(item2)
    assert len(history) == 2
    _assert_current(item2)
    assert history.forward() is None
    assert history.back() is not None
    assert history.back() is None


def test_iter_history():
    history = History()

    # Wrong arguments to iter().
    assert len([_ for _ in history.iter(0, 0)]) == 0
    assert len([_ for _ in history.iter(2, 1)]) == 0

    item0 = np.zeros(3)
    item1 = np.ones(4)
    item2 = 2 * np.ones(5)

    history.add(item0)
    history.add(item1)
    history.add(item2)

    for i, item in enumerate(history):
        # Assert item<i>
        if i > 0:
            assert id(item) == id(locals()['item{0:d}'.format(i - 1)])

    for i, item in enumerate(history.iter(1, 2)):
        assert i == 0
        # Assert item<i>
        assert history.current_position == 3
        assert id(item) == id(locals()['item{0:d}'.format(i)])

    for i, item in enumerate(history.iter(2, 3)):
        assert i == 0
        # Assert item<i>
        assert history.current_position == 3
        assert id(item) == id(locals()['item{0:d}'.format(i + 1)])


def test_global_history():
    gh = GlobalHistory()

    h1 = History()
    h2 = History()

    # First action.
    h1.add('h1 first')
    gh.action(h1)

    # Second action.
    h2.add('h2 first')
    gh.action(h2)

    # Third action.
    h1.add('h1 second')
    h2.add('h2 second')
    gh.action(h1, h2)

    # Fourth action.
    h1.add('h1 third')
    gh.action(h1)

    # Undo once.
    assert gh.undo() == ('h1 third',)

    # Undo once more.
    assert gh.undo() == ('h1 second', 'h2 second')

    # Redo once.
    assert gh.redo() == ('h1 second', 'h2 second')

    # New fourth action.
    h1.add('h1 third')
    h2.add('h2 third')
    gh.action(h1)
    gh.add_to_current_action(h2)

    assert gh.redo() == ()
    assert gh.undo() == ('h1 third', 'h2 third')
    assert gh.undo() == ('h1 second', 'h2 second')
    assert gh.undo() == ('h2 first',)
    assert gh.undo() == ('h1 first',)

    gh.process_ups = lambda ups: ''.join(ups)

    assert gh.undo() == ''
    assert gh.undo() == ''
    assert gh.redo() == 'h1 first'
    assert gh.redo() == 'h2 first'
    assert gh.redo() == 'h1 second' + 'h2 second'
