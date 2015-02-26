# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ..session import Session, start_manual_clustering
from ....utils.tempdir import TemporaryDirectory
from ....io.mock.artificial import MockModel
from ....io.mock.kwik import create_mock_kwik


#------------------------------------------------------------------------------
# Generic tests
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
        _track.append('my event')

    assert _track == []

    session.emit('invalid')
    assert _track == []

    session.emit('my_event')
    assert _track == ['my event']

    # Although the callback doesn't accept a 'data' keyword argument, this does
    # not raise an error because the event system will only pass the argument
    # if it is part of the callback arg spec.
    session.emit('my_event', data='hello')


def test_session_connect_multiple():
    """Test @connect decorator and event system."""
    session = Session()

    _track = []

    @session.connect
    def on_my_event():
        _track.append('my event')

    @session.connect
    def on_my_event():
        _track.append('my event again')

    session.emit('my_event')
    assert _track == ['my event', 'my event again']


def test_session_unconnect():
    """Test unconnect."""
    session = Session()

    _track = []

    @session.connect
    def on_my_event():
        _track.append('my event')

    session.emit('my_event')
    assert _track == ['my event']

    # Unregister and test that the on_my_event() callback is no longer called.
    session.unconnect(on_my_event)
    session.emit('my_event')
    assert _track == ['my event']


def test_session_connect_alternative():
    """Test the alternative @connect() syntax."""
    session = Session()

    _track = []

    assert _track == []

    @session.connect()
    def on_my_event():
        _track.append('my event')

    session.emit('my_event')
    assert _track == ['my event']


def test_action():
    session = Session()
    _track = []

    @session.action(title='My action')
    def my_action():
        _track.append('action')

    session.my_action()
    assert _track == ['action']


def test_action_event():
    session = Session()
    _track = []

    @session.connect
    def on_hello(out, kwarg=''):
        _track.append(out + kwarg)

    # We forgot the 'title=', but this still works.
    @session.action('My action')
    def my_action_hello(data):
        _track.append(data)
        session.emit('hello', data + ' world', kwarg='!')

    # Need one argument.
    with raises(TypeError):
        session.my_action_hello()

    # This triggers the 'hello' event which adds 'hello world' to _track.
    session.my_action_hello('hello')
    assert _track == ['hello', 'hello world!']


#------------------------------------------------------------------------------
# Kwik tests
#------------------------------------------------------------------------------

def test_session_mock():
    session = start_manual_clustering(model=MockModel())
    view = session.show_waveforms()
    session.select([0])
    view_bis = session.show_waveforms()

    session.merge([3, 4])

    view.close()
    view_bis.close()

    session = start_manual_clustering(model=MockModel())
    session.select([1, 2])
    view = session.show_waveforms()
    view.close()


def test_session_kwik():

    n_clusters = 5
    n_spikes = 50
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=n_clusters,
                                    n_spikes=n_spikes,
                                    n_channels=n_channels,
                                    n_features_per_channel=n_fets,
                                    n_samples_traces=n_samples_traces)

        session = start_manual_clustering(filename)
        session.select([0])
        session.merge([3, 4])
        view = session.show_waveforms()

        # This won't work but shouldn't raise an error.
        session.select([1000])

        # TODO: more tests
        session.undo()
        session.redo()

        view.close()


def test_session_stats():

    n_clusters = 5
    n_spikes = 50
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=n_clusters,
                                    n_spikes=n_spikes,
                                    n_channels=n_channels,
                                    n_features_per_channel=n_fets,
                                    n_samples_traces=n_samples_traces)

        session = start_manual_clustering(filename)
        assert session

        # TODO

        # masks = session.stats.cluster_masks(3)
        # assert masks.shape == (n_channels,)

        # session.merge([3, 4])

        # masks = session.stats.cluster_masks(3)
        # assert masks.shape == (n_channels,)

        # masks = session.stats.cluster_masks(n_clusters)
        # assert masks.shape == (n_channels,)
