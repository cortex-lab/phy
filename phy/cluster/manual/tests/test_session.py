# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_allclose as ac
from pytest import raises

from ..session import BaseSession, Session, FeatureMasks
from ....utils.testing import show_test
from ....utils.tempdir import TemporaryDirectory
from ....utils.logging import set_level
from ....io.mock.artificial import MockModel
from ....io.mock.kwik import create_mock_kwik


#------------------------------------------------------------------------------
# Generic tests
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def test_session_connect():
    """Test @connect decorator and event system."""
    session = BaseSession()

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
    session = BaseSession()

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
    session = BaseSession()

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
    session = BaseSession()

    _track = []

    assert _track == []

    @session.connect()
    def on_my_event():
        _track.append('my event')

    session.emit('my_event')
    assert _track == ['my event']


def test_action():
    session = BaseSession()
    _track = []

    @session.action(title='My action')
    def my_action():
        _track.append('action')

    session.my_action()
    assert _track == ['action']

    assert session.actions == [{'func': my_action, 'title': 'My action'}]
    session.execute_action(session.actions[0])
    assert _track == ['action', 'action']


def test_action_event():
    session = BaseSession()
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

def _start_manual_clustering(filename=None, model=None, tempdir=None):
    session = Session(phy_user_dir=tempdir)
    session.open(filename=filename, model=model)
    return session


def _show_view(session, name):
    if name == 'waveforms':
        vm = session._create_waveform_view_model()
    elif name == 'features':
        vm = session._create_feature_view_model()
    elif name == 'correlograms':
        vm = session._create_correlogram_view_model()
    vm.scale_factor = 1.
    view = session._create_view(vm, show=False)
    show_test(view)
    return view


def test_session_store():
    """Check that the cluster store works for features and masks."""

    # HACK: change the chunk size in this unit test to make sure that
    # there are several chunks.
    cs = FeatureMasks.chunk_size
    FeatureMasks.chunk_size = 4

    with TemporaryDirectory() as tempdir:
        model = MockModel(n_spikes=10, n_clusters=3)
        s0 = np.nonzero(model.spike_clusters == 0)[0]
        s1 = np.nonzero(model.spike_clusters == 1)[0]

        session = _start_manual_clustering(model=model,
                                           tempdir=tempdir)

        f = session.cluster_store.features(0)
        m = session.cluster_store.masks(1)

        assert f.shape == (len(s0), 28, 2)
        assert m.shape == (len(s1), 28,)

        ac(f, model.features[s0].reshape((f.shape[0], -1, 2)), 1e-3)
        ac(m, model.masks[s1], 1e-3)

    FeatureMasks.chunk_size = cs


def test_session_mock():
    with TemporaryDirectory() as tempdir:
        session = _start_manual_clustering(model=MockModel(),
                                           tempdir=tempdir)

        view = _show_view(session, 'waveforms')
        session.select([0])
        view_bis = _show_view(session, 'waveforms')

        view.close()
        view_bis.close()

        session = _start_manual_clustering(model=MockModel(), tempdir=tempdir)
        session.select([1, 2])
        view = _show_view(session, 'waveforms')

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

        session = _start_manual_clustering(filename=filename,
                                           tempdir=tempdir)

        session.select([0])
        cs = session.cluster_store

        # Check the stored items.
        for cluster in range(n_clusters):
            n_spikes = len(session.clustering.spikes_per_cluster[cluster])
            n_unmasked_channels = cs.n_unmasked_channels(cluster)

            assert cs.features(cluster).shape == (n_spikes, n_channels, n_fets)
            assert cs.masks(cluster).shape == (n_spikes, n_channels)
            assert cs.mean_masks(cluster).shape == (n_channels,)
            assert n_unmasked_channels <= n_channels
            assert cs.mean_probe_position(cluster).shape == (2,)
            assert cs.main_channels(cluster).shape == (n_unmasked_channels,)

        view_0 = _show_view(session, 'waveforms')
        view_1 = _show_view(session, 'features')

        # This won't work but shouldn't raise an error.
        session.select([1000])

        view_0.close()
        view_1.close()
        session.close()
