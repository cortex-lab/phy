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


def test_session_connect_alternative():
    session = Session()

    _track = []

    assert _track == []

    @session.connect()
    def on_my_event():
        _track.append("my event")

    assert _track == ["my event"]
