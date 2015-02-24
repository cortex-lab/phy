# -*- coding: utf-8 -*-

"""Test event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..event import EventEmitter


#------------------------------------------------------------------------------
# Test event system
#------------------------------------------------------------------------------

def test_event_system():
    ev = EventEmitter()

    _list = []

    @ev.connect
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
