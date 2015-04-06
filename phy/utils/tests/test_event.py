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
    def on_report(value, value_max):
        # value is the sum of the values, value_max the sum of the max values
        _reported.append((value, value_max))

    @pr.connect
    def on_complete():
        _completed.append(True)

    pr.set_max(channel_1=10, channel_2=15)
    assert _reported == []
    assert pr.current() == 0
    assert pr.total() == 25

    pr.set(channel_1=7)
    assert _reported == [(7, 25)]
    assert pr.current() == 7
    assert pr.total() == 25

    with raises(ValueError):
        pr.set(channel_1=11)

    with raises(ValueError):
        pr.set_max(channel_1=6)

    pr.set(channel_2=13)
    assert _reported[-1] == (20, 25)
    assert pr.current() == 20
    assert pr.total() == 25

    pr.increment('channel_1', 'channel_2')
    assert _reported[-1] == (22, 25)
    assert pr.current() == 22
    assert pr.total() == 25

    pr.set(channel_1=10, channel_2=15)
    assert _reported[-1] == (25, 25)
    assert _completed == [True]
    assert pr.is_complete()

    pr.set_max(channel_2=20)
    assert not pr.is_complete()
    pr.set(channel_1=10, channel_2=20)
    assert pr.is_complete()
