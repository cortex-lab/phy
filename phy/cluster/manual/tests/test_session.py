# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from pytest import raises

from ..session import Session


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_session_connect():
    """Test @connect decorator and event system."""
    session = Session()

    # connect names should be on_something().
    with raises(ValueError):
        @session.connect
        def invalid():
            pass

    _track = []

    @session.connect
    def on_my_event():
        _track.append("my event")

    assert _track == []

    session.emit("invalid")
    assert _track == []

    session.emit("my_event")
    assert _track == ["my event"]


def test_session_unconnect():
    """Test unconnect."""
    session = Session()

    _track = []

    @session.connect
    def on_my_event():
        _track.append("my event")

    session.emit("my_event")
    assert _track == ["my event"]

    # Unregister and test that the on_my_event() callback is no longer called.
    session.unconnect(on_my_event)
    session.emit("my_event")
    assert _track == ["my event"]


def test_session_connect_alternative():
    """Test the alternative @connect() syntax."""
    session = Session()

    _track = []

    assert _track == []

    @session.connect()
    def on_my_event():
        _track.append("my event")

    session.emit("my_event")
    assert _track == ["my event"]


def test_action():
    session = Session()
    _track = []

    @session.action(title="My action")
    def my_action():
        _track.append("action")

    session.my_action()
    assert _track == ["action"]


def test_action_event():
    session = Session()
    _track = []

    @session.connect
    def on_hello(out):
        _track.append(out)

    @session.action(title="My action", event="hello")
    def my_action_hello(data):
        _track.append(data)
        return data + " world"

    # Need one argument.
    with raises(TypeError):
        session.my_action_hello()

    # This triggers the 'hello' event which adds 'hello world' to _track.
    session.my_action_hello("hello")
    assert _track == ["hello", "hello world"]
