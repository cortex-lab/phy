# -*- coding: utf-8 -*-

"""Test event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..event import EventEmitter, ProgressReporter


#------------------------------------------------------------------------------
# Test event system
#------------------------------------------------------------------------------

def test_event_system():
    ev = EventEmitter()

    _list = []

    with raises(ValueError):
        ev.connect(lambda x: x)

    @ev.connect(set_method=True)
    def on_my_event(arg, kwarg=None):
        _list.append((arg, kwarg))

    with raises(TypeError):
        ev.my_event()

    ev.my_event('a')
    assert _list == [('a', None)]

    ev.my_event('b', 'c')
    assert _list == [('a', None), ('b', 'c')]

    ev.unconnect(on_my_event)

    ev.my_event('b', 'c')
    assert _list == [('a', None), ('b', 'c')]


#------------------------------------------------------------------------------
# Test progress reporter
#------------------------------------------------------------------------------

def test_progress_reporter():
    """Test the progress reporter."""
    pr = ProgressReporter()

    _reported = []
    _completed = []

    @pr.connect
    def on_progress(value, value_max):
        # value is the sum of the values, value_max the sum of the max values
        _reported.append((value, value_max))

    @pr.connect
    def on_complete():
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
