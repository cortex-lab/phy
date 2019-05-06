# -*- coding: utf-8 -*-

"""Test event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..event import EventEmitter, ProgressReporter, connect


#------------------------------------------------------------------------------
# Test event system
#------------------------------------------------------------------------------

def test_event_system():
    ev = EventEmitter()

    _list = []

    with raises(ValueError):
        ev.connect(lambda x: x)

    @ev.connect
    def on_my_event(sender, arg, kwarg=None):
        _list.append((arg, kwarg))

    ev.emit('my_event', ev, 'a')
    assert _list == [('a', None)]

    ev.emit('my_event', ev, 'b', 'c')
    assert _list == [('a', None), ('b', 'c')]

    ev.unconnect(on_my_event)

    ev.emit('my_event', ev, 'b', 'c')
    assert _list == [('a', None), ('b', 'c')]


def test_event_silent():
    ev = EventEmitter()

    _list = []

    @ev.connect()
    def on_test(sender, x):
        _list.append(x)

    ev.emit('test', ev, 1)
    assert _list == [1]

    with ev.silent():
        ev.emit('test', ev, 1)
    assert _list == [1]


def test_event_single():
    ev = EventEmitter()

    l = []

    @ev.connect
    def on_test(sender):
        l.append(0)

    @ev.connect  # noqa
    def on_test(sender):
        l.append(1)

    ev.emit('test', ev)
    assert l == [0, 1]

    ev.emit('test', ev, single=True)
    assert l == [0, 1, 0]


#------------------------------------------------------------------------------
# Test progress reporter
#------------------------------------------------------------------------------

def test_progress_reporter():
    """Test the progress reporter."""
    pr = ProgressReporter()

    _reported = []
    _completed = []

    @connect(sender=pr)
    def on_progress(sender, value, value_max):
        # value is the sum of the values, value_max the sum of the max values
        _reported.append((value, value_max))

    @connect(sender=pr)
    def on_complete(sender):
        _completed.append(True)

    pr.value_max = 10
    pr.value = 0
    pr.value = 5
    assert pr.value == 5
    assert pr.progress == .5
    assert not pr.is_complete()
    pr.value = 10
    assert pr.is_complete()
    assert pr.progress == 1.
    assert _completed == [True]

    pr.value_max = 11
    assert not pr.is_complete()
    assert pr.progress < 1.
    pr.set_complete()
    assert pr.is_complete()
    assert pr.progress == 1.

    assert _reported == [(0, 10), (5, 10), (10, 10), (11, 11)]
    assert _completed == [True, True]

    pr.value = 10
    # Only trigger a complete event once.
    pr.value = pr.value_max
    pr.value = pr.value_max
    assert _completed == [True, True, True]


def test_progress_message():
    """Test messages with the progress reporter."""
    pr = ProgressReporter()
    pr.reset(5)
    pr.set_progress_message("The progress is {progress}%. ({hello:d})")
    pr.set_complete_message("Finished {hello}.")

    pr.value_max = 10
    pr.value = 0
    print()
    pr.value = 5
    print()
    pr.increment()
    print()
    pr.increment(hello='hello')
    print()
    pr.increment(hello=3)
    print()
    pr.value = 10
