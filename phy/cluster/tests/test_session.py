# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_allclose as ac
from pytest import raises, fixture

from ..session import Session
from ...utils import _spikes_in_clusters
from ...gui.qt import wrap_qt
from ...utils.tempdir import TemporaryDirectory
from ...utils.logging import set_level
from ...io.mock import MockModel
from ...io.kwik.mock import create_mock_kwik


#------------------------------------------------------------------------------
# Kwik tests
#------------------------------------------------------------------------------

def setup():
    set_level('info')


def _start_manual_clustering(kwik_path=None,
                             model=None,
                             tempdir=None,
                             chunk_size=None,
                             ):
    session = Session(phy_user_dir=tempdir)
    if chunk_size is not None:
        session.settings['features_masks_chunk_size'] = chunk_size
    session.open(kwik_path=kwik_path, model=model)
    return session


def test_session_store_features():
    """Check that the cluster store works for features and masks."""

    with TemporaryDirectory() as tempdir:
        model = MockModel(n_spikes=50, n_clusters=3)
        s0 = np.nonzero(model.spike_clusters == 0)[0]
        s1 = np.nonzero(model.spike_clusters == 1)[0]

        session = _start_manual_clustering(model=model,
                                           tempdir=tempdir,
                                           chunk_size=4,
                                           )

        f = session.cluster_store.features(0)
        m = session.cluster_store.masks(1)
        w = session.cluster_store.waveforms(1)

        assert f.shape == (len(s0), 28, 2)
        assert m.shape == (len(s1), 28,)
        assert w.shape == (len(s1), model.n_samples_waveforms, 28,)

        ac(f, model.features[s0].reshape((f.shape[0], -1, 2)), 1e-3)
        ac(m, model.masks[s1], 1e-3)


n_clusters = 5
n_spikes = 50
n_channels = 28
n_fets = 2
n_samples_traces = 3000


@fixture
def session(request):
    tmpdir = TemporaryDirectory()

    # Create the test HDF5 file in the temporary directory.
    kwik_path = create_mock_kwik(tmpdir.name,
                                 n_clusters=n_clusters,
                                 n_spikes=n_spikes,
                                 n_channels=n_channels,
                                 n_features_per_channel=n_fets,
                                 n_samples_traces=n_samples_traces)

    session = _start_manual_clustering(kwik_path=kwik_path,
                                       tempdir=tmpdir.name)
    session.tempdir = tmpdir.name

    def end():
        session.close()
        tmpdir.cleanup()
    request.addfinalizer(end)

    return session


@wrap_qt
def test_session_clustering(session):

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

    gui = session.show_gui()
    yield

    # Merge two clusters.
    clusters = [0, 2]
    gui.merge(clusters)  # Create cluster 5.
    _check_arrays(5, clusters)
    yield

    # Split some spikes.
    spikes = [2, 3, 5, 7, 11, 13]
    # clusters = np.unique(spike_clusters[spikes])
    gui.split(spikes)  # Create cluster 6 and more.
    _check_arrays(6, spikes=spikes)
    yield

    # Undo.
    gui.undo()
    _check_arrays(5, clusters)
    yield

    # Undo.
    gui.undo()
    _check_arrays(0)
    _check_arrays(2)
    yield

    # Redo.
    gui.redo()
    _check_arrays(5, clusters)
    yield

    # Split some spikes.
    spikes = [5, 7, 11, 13, 17, 19]
    # clusters = np.unique(spike_clusters[spikes])
    gui.split(spikes)  # Create cluster 6 and more.
    _check_arrays(6, spikes=spikes)
    yield

    # Test merge-undo-different-merge combo.
    spc = gui.clustering.spikes_per_cluster.copy()
    clusters = gui.cluster_ids[:3]
    up = gui.merge(clusters)
    _check_arrays(up.added[0], spikes=up.spike_ids)
    # Undo.
    gui.undo()
    for cluster in clusters:
        _check_arrays(cluster, spikes=spc[cluster])
    # Another merge.
    clusters = gui.cluster_ids[1:5]
    up = gui.merge(clusters)
    _check_arrays(up.added[0], spikes=up.spike_ids)
    yield

    # Move a cluster to a group.
    cluster = gui.cluster_ids[0]
    gui.move([cluster], 2)
    assert len(gui.store.mean_probe_position(cluster)) == 2
    yield

    # Save.
    spike_clusters_new = gui.model.spike_clusters.copy()
    # Check that the spike clusters have changed.
    assert not np.all(spike_clusters_new == spike_clusters)
    ac(session.model.spike_clusters, gui.clustering.spike_clusters)
    session.save()
    yield

    # Re-open the file and check that the spike clusters and
    # cluster groups have correctly been saved.
    session = _start_manual_clustering(kwik_path=session.model.path,
                                       tempdir=session.tempdir)
    ac(session.model.spike_clusters, gui.clustering.spike_clusters)
    ac(session.model.spike_clusters, spike_clusters_new)
    #  Check the cluster groups.
    clusters = gui.clustering.cluster_ids
    groups = session.model.cluster_groups
    assert groups[cluster] == 2
    yield

    gui.close()


@wrap_qt
def test_session_multiple_clusterings(session):

    gui = session.show_gui()
    yield

    assert session.model.n_spikes == n_spikes
    assert session.model.n_clusters == n_clusters
    assert len(session.model.cluster_ids) == n_clusters
    assert gui.clustering.n_clusters == n_clusters
    assert session.model.cluster_metadata.group(1) == 1

    # Change clustering.
    with raises(ValueError):
        session.change_clustering('automat')
    session.change_clustering('automatic')
    yield

    n_clusters_2 = session.model.n_clusters
    assert session.model.n_spikes == n_spikes
    assert session.model.n_clusters == n_clusters_2
    assert len(session.model.cluster_ids) == n_clusters_2
    assert gui.clustering.n_clusters == n_clusters_2
    assert session.model.cluster_metadata.group(2) == 2

    # Merge the clusters and save, for the current clustering.
    gui.clustering.merge(gui.clustering.cluster_ids)
    session.save()
    yield

    # Re-open the session.
    session = _start_manual_clustering(kwik_path=session.model.path,
                                       tempdir=session.tempdir)
    yield

    # The default clustering is the main one: nothing should have
    # changed here.
    assert session.model.n_clusters == n_clusters

    session.change_clustering('automatic')
    assert session.model.n_spikes == n_spikes
    assert session.model.n_clusters == 1
    assert session.model.cluster_ids == n_clusters_2
    yield

    gui.close()


def test_session_kwik(session):

    # Check backup.
    assert op.exists(op.join(session.tempdir, session.kwik_path + '.bak'))

    cs = session.cluster_store
    nc = n_channels - 2

    # Check the stored items.
    for cluster in range(n_clusters):
        n_spikes = len(session.model.spikes_per_cluster[cluster])
        n_unmasked_channels = cs.n_unmasked_channels(cluster)

        assert cs.features(cluster).shape == (n_spikes, nc, n_fets)
        assert cs.masks(cluster).shape == (n_spikes, nc)
        assert cs.mean_masks(cluster).shape == (nc,)
        assert n_unmasked_channels <= nc
        assert cs.mean_probe_position(cluster).shape == (2,)
        assert cs.main_channels(cluster).shape == (n_unmasked_channels,)
