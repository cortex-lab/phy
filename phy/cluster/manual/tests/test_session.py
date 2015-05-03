# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_allclose as ac
from numpy.testing import assert_array_equal as ae
from pytest import raises

from .._utils import _spikes_in_clusters
from ..session import BaseSession, Session, FeatureMasks
from ....utils.testing import show_test
from ....utils.dock import qt_app, _close_qt_after
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

def _start_manual_clustering(kwik_path=None, model=None, tempdir=None):
    session = Session(phy_user_dir=tempdir)
    session.open(kwik_path=kwik_path, model=model)
    return session


def _show_view(session, name):
    vm = session.create_view(name)
    vm.scale_factor = 1.
    show_test(vm.view)
    return vm.view


def test_session_store():
    """Check that the cluster store works for features and masks."""

    # HACK: change the chunk size in this unit test to make sure that
    # there are several chunks.
    cs = FeatureMasks.chunk_size
    FeatureMasks.chunk_size = 4

    with TemporaryDirectory() as tempdir:
        model = MockModel(n_spikes=50, n_clusters=3)
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
        view.close()


def test_session_gui():
    n_clusters = 5
    n_spikes = 50
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)

        with qt_app():
            gui = session._create_gui()
            _close_qt_after(gui, 0.2)
            gui.show()


def test_session_kwik():
    n_clusters = 5
    n_spikes = 50
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)

        # Check backup.
        assert op.exists(op.join(tempdir, kwik_path + '.bak'))

        session.select([0])
        cs = session.cluster_store

        nc = n_channels - 2

        # Check the stored items.
        for cluster in range(n_clusters):
            n_spikes = len(session.clustering.spikes_per_cluster[cluster])
            n_unmasked_channels = cs.n_unmasked_channels(cluster)

            assert cs.features(cluster).shape == (n_spikes, nc, n_fets)
            assert cs.masks(cluster).shape == (n_spikes, nc)
            assert cs.mean_masks(cluster).shape == (nc,)
            assert n_unmasked_channels <= nc
            assert cs.mean_probe_position(cluster).shape == (2,)
            assert cs.main_channels(cluster).shape == (n_unmasked_channels,)

        view_0 = _show_view(session, 'waveforms')
        view_1 = _show_view(session, 'features')

        # This won't work but shouldn't raise an error.
        session.select([1000])

        view_0.close()
        view_1.close()
        session.close()


def test_session_clustering():

    n_clusters = 5
    n_spikes = 50
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)
        cs = session.cluster_store
        spike_clusters = session.model.spike_clusters.copy()

        f = session.model.features
        m = session.model.masks

        def _check_arrays(cluster, clusters_for_sc=None, spikes=None):
            """Check the features and masks in the cluster store
            of a given custer."""
            if spikes is None:
                if clusters_for_sc is None:
                    clusters_for_sc = [cluster]
                spikes = _spikes_in_clusters(spike_clusters, clusters_for_sc)
            shape = (len(spikes),
                     len(session.model.channel_order),
                     session.model.n_features_per_channel)
            ac(cs.features(cluster), f[spikes, :].reshape(shape))
            ac(cs.masks(cluster), m[spikes])

        _check_arrays(0)
        _check_arrays(2)

        # Test session.best_clusters.
        quality = session.cluster_store.n_unmasked_channels
        clusters = session.best_clusters(quality)
        ac(np.unique(clusters), session.cluster_ids)

        # Merge two clusters.
        clusters = [0, 2]
        session.merge(clusters)  # Create cluster 5.
        _check_arrays(5, clusters)

        # Split some spikes.
        spikes = [2, 3, 5, 7, 11, 13]
        # clusters = np.unique(spike_clusters[spikes])
        session.split(spikes)  # Create cluster 6 and more.
        _check_arrays(6, spikes=spikes)

        # Undo.
        session.undo()
        _check_arrays(5, clusters)

        # Undo.
        session.undo()
        _check_arrays(0)
        _check_arrays(2)

        # Redo.
        session.redo()
        _check_arrays(5, clusters)

        # Split some spikes.
        spikes = [5, 7, 11, 13, 17, 19]
        # clusters = np.unique(spike_clusters[spikes])
        session.split(spikes)  # Create cluster 6 and more.
        _check_arrays(6, spikes=spikes)

        # Move a cluster to a group.
        session.move([6], 2)
        assert len(session.cluster_store.mean_probe_position(6)) == 2

        # Save.
        spike_clusters_new = session.model.spike_clusters.copy()
        # Check that the spike clusters have changed.
        assert not np.all(spike_clusters_new == spike_clusters)
        ac(session.model.spike_clusters, session.clustering.spike_clusters)
        session.save()

        # Re-open the file and check that the spike clusters and
        # cluster groups have correctly been saved.
        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)
        ac(session.model.spike_clusters, session.clustering.spike_clusters)
        ac(session.model.spike_clusters, spike_clusters_new)
        #  Check the cluster groups.
        clusters = session.clustering.cluster_ids
        groups = session.model.cluster_groups
        assert groups[6] == 2


def test_session_wizard():

    n_clusters = 5
    n_spikes = 100
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)

        clusters = np.arange(n_clusters)
        best_clusters = session.best_clusters()

        assert session.wizard.best_cluster() == best_clusters[0]
        ae(np.unique(best_clusters), clusters)
        assert len(session.wizard.most_similar_clusters()) == n_clusters - 1

        assert len(session.wizard.most_similar_clusters(0, n_max=3)) == 3

        session.merge([0, 1, 2])
        ae(np.unique(session.best_clusters()), np.arange(3, 6))
        assert list(session.wizard.most_similar_clusters(5)) in ([3, 4],
                                                                 [4, 3])

        # Move a cluster to noise.
        session.move([5], 0)
        ae(np.unique(session.best_clusters()), np.arange(3, 5))
        best = session.wizard.best_cluster()
        if best is not None:
            assert best in (3, 4)
            # The most similar cluster is 3 if best=4 and conversely.
            assert list(session.wizard.most_similar_clusters(best)) == [7 -
                                                                        best]


def test_session_multiple_clusterings():

    n_clusters = 5
    n_spikes = 100
    n_channels = 28
    n_fets = 2
    n_samples_traces = 10000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)

        assert session.model.n_spikes == n_spikes
        assert session.model.n_clusters == n_clusters
        assert len(session.model.cluster_ids) == n_clusters
        assert session.clustering.n_clusters == n_clusters
        assert session.cluster_metadata.group(1) == 3

        session.select([0, 1])

        # Change clustering.
        with raises(ValueError):
            session.change_clustering('automat')
        session.change_clustering('automatic')

        assert session.model.n_spikes == n_spikes
        assert session.model.n_clusters == n_clusters * 2
        assert len(session.model.cluster_ids) == n_clusters * 2
        assert session.clustering.n_clusters == n_clusters * 2
        assert session.cluster_metadata.group(2) == 3

        # The current selection is cleared when changing clustering.
        ae(session._selected_clusters, [])

        # Merge the clusters and save, for the current clustering.
        session.clustering.merge(session.clustering.cluster_ids)
        session.save()
        session.close()

        # Re-open the session.
        session = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)
        # The default clustering is the main one: nothing should have
        # changed here.
        assert session.model.n_clusters == n_clusters
        session.change_clustering('automatic')
        assert session.model.n_spikes == n_spikes
        assert session.model.n_clusters == 1
        assert session.model.cluster_ids == n_clusters * 2
