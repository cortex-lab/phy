# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

import numpy as np
from numpy.testing import assert_allclose as ac
from numpy.testing import assert_equal as ae
from pytest import raises, yield_fixture, mark

from ..session import Session
from ...utils.array import _spikes_in_clusters
from ...utils.logging import set_level
from ...io.mock import MockModel, artificial_traces
from ...io.kwik.mock import create_mock_kwik
from ...io.kwik.creator import create_kwik


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    set_level('info')


n_clusters = 5
n_spikes = 50
n_channels = 28
n_fets = 2
n_samples_traces = 3000


def _start_manual_clustering(kwik_path=None,
                             model=None,
                             tempdir=None,
                             chunk_size=None,
                             ):
    session = Session(phy_user_dir=tempdir)
    session.open(kwik_path=kwik_path, model=model)
    session.settings['waveforms_scale_factor'] = 1.
    session.settings['features_scale_factor'] = 1.
    session.settings['traces_scale_factor'] = 1.
    session.settings['prompt_save_on_exit'] = False
    if chunk_size is not None:
        session.settings['features_masks_chunk_size'] = chunk_size
    return session


@yield_fixture
def session(tempdir):
    # Create the test HDF5 file in the temporary directory.
    kwik_path = create_mock_kwik(tempdir,
                                 n_clusters=n_clusters,
                                 n_spikes=n_spikes,
                                 n_channels=n_channels,
                                 n_features_per_channel=n_fets,
                                 n_samples_traces=n_samples_traces)

    session = _start_manual_clustering(kwik_path=kwik_path,
                                       tempdir=tempdir)
    session.tempdir = tempdir

    yield session

    session.close()


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_store_corruption(tempdir):
    # Create the test HDF5 file in the temporary directory.
    kwik_path = create_mock_kwik(tempdir,
                                 n_clusters=n_clusters,
                                 n_spikes=n_spikes,
                                 n_channels=n_channels,
                                 n_features_per_channel=n_fets,
                                 n_samples_traces=n_samples_traces)

    session = Session(kwik_path, phy_user_dir=tempdir)
    store_path = session.store.path
    session.close()

    # Corrupt a file in the store.
    fn = op.join(store_path, 'waveforms_spikes')
    with open(fn, 'rb') as f:
        contents = f.read()
    with open(fn, 'wb') as f:
        f.write(contents[1:-1])

    session = Session(kwik_path, phy_user_dir=tempdir)
    session.close()


def test_session_one_cluster(tempdir):
    session = Session(phy_user_dir=tempdir)
    # The disk store is not created if there is only one cluster.
    session.open(model=MockModel(n_clusters=1))
    assert session.store.disk_store is None


def test_session_store_features(tempdir):
    """Check that the cluster store works for features and masks."""

    model = MockModel(n_spikes=50, n_clusters=3)
    s0 = np.nonzero(model.spike_clusters == 0)[0]
    s1 = np.nonzero(model.spike_clusters == 1)[0]

    session = _start_manual_clustering(model=model,
                                       tempdir=tempdir,
                                       chunk_size=4,
                                       )

    f = session.store.features(0)
    m = session.store.masks(1)
    w = session.store.waveforms(1)

    assert f.shape == (len(s0), 28, 2)
    assert m.shape == (len(s1), 28,)
    assert w.shape == (len(s1), model.n_samples_waveforms, 28,)

    ac(f, model.features[s0].reshape((f.shape[0], -1, 2)), 1e-3)
    ac(m, model.masks[s1], 1e-3)


def test_session_gui_clustering(qtbot, session):

    cs = session.store
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
    qtbot.addWidget(gui.main_window)

    # Merge two clusters.
    clusters = [0, 2]
    gui.merge(clusters)  # Create cluster 5.
    _check_arrays(5, clusters)

    # Split some spikes.
    spikes = [2, 3, 5, 7, 11, 13]
    # clusters = np.unique(spike_clusters[spikes])
    gui.split(spikes)  # Create cluster 6 and more.
    _check_arrays(6, spikes=spikes)

    # Undo.
    gui.undo()
    _check_arrays(5, clusters)

    # Undo.
    gui.undo()
    _check_arrays(0)
    _check_arrays(2)

    # Redo.
    gui.redo()
    _check_arrays(5, clusters)

    # Split some spikes.
    spikes = [5, 7, 11, 13, 17, 19]
    # clusters = np.unique(spike_clusters[spikes])
    gui.split(spikes)  # Create cluster 6 and more.
    _check_arrays(6, spikes=spikes)

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

    # Move a cluster to a group.
    cluster = gui.cluster_ids[0]
    gui.move([cluster], 2)
    assert len(gui.store.mean_probe_position(cluster)) == 2

    # Save.
    spike_clusters_new = gui.model.spike_clusters.copy()
    # Check that the spike clusters have changed.
    assert not np.all(spike_clusters_new == spike_clusters)
    ac(session.model.spike_clusters, gui.clustering.spike_clusters)
    session.save()

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

    gui.close()


def test_session_gui_multiple_clusterings(qtbot, session):

    gui = session.show_gui()
    qtbot.addWidget(gui.main_window)

    assert session.model.n_spikes == n_spikes
    assert session.model.n_clusters == n_clusters
    assert len(session.model.cluster_ids) == n_clusters
    assert gui.clustering.n_clusters == n_clusters
    assert session.model.cluster_metadata.group(1) == 1

    # Change clustering.
    with raises(ValueError):
        session.change_clustering('automat')
    session.change_clustering('original')

    n_clusters_2 = session.model.n_clusters
    assert session.model.n_spikes == n_spikes
    assert session.model.n_clusters == n_clusters_2
    assert len(session.model.cluster_ids) == n_clusters_2
    assert gui.clustering.n_clusters == n_clusters_2
    assert session.model.cluster_metadata.group(2) == 2

    # Merge the clusters and save, for the current clustering.
    gui.clustering.merge(gui.clustering.cluster_ids)
    session.save()

    # Re-open the session.
    session = _start_manual_clustering(kwik_path=session.model.path,
                                       tempdir=session.tempdir)

    # The default clustering is the main one: nothing should have
    # changed here.
    assert session.model.n_clusters == n_clusters

    session.change_clustering('original')
    assert session.model.n_spikes == n_spikes
    assert session.model.n_clusters == 1
    assert session.model.cluster_ids == n_clusters_2

    gui.close()


def test_session_kwik(session):

    # Check backup.
    assert op.exists(op.join(session.tempdir, session.kwik_path + '.bak'))

    cs = session.store
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


def test_session_gui_statistics(qtbot, session):
    """Test registration of new statistic."""

    gui = session.show_gui()
    qtbot.addWidget(gui.main_window)

    @session.register_statistic
    def n_spikes_2(cluster):
        return gui.clustering.cluster_counts.get(cluster, 0) ** 2

    store = gui.store
    stats = store.items['statistics']

    def _check():
        for clu in gui.cluster_ids:
            assert (store.n_spikes_2(clu) ==
                    store.features(clu).shape[0] ** 2)

    assert 'n_spikes_2' in stats.fields
    _check()

    # Merge the clusters and check that the statistics has been
    # recomputed for the new cluster.
    clusters = gui.cluster_ids
    gui.merge(clusters)
    _check()
    assert gui.cluster_ids == [max(clusters) + 1]

    gui.undo()
    _check()

    gui.merge(gui.cluster_ids[::2])
    _check()

    gui.close()


@mark.parametrize('spike_ids', [None, np.arange(20)])
def test_session_auto(session, spike_ids):
    set_level('info')
    sc = session.cluster(num_starting_clusters=10,
                         spike_ids=spike_ids,
                         clustering='test',
                         )
    assert session.model.clustering == 'test'
    assert session.model.clusterings == ['main',
                                         'original', 'test']

    # Re-open the dataset and check that the clustering has been saved.
    session = _start_manual_clustering(kwik_path=session.model.path,
                                       tempdir=session.tempdir)
    if spike_ids is None:
        spike_ids = slice(None, None, None)
    assert len(session.model.spike_clusters) == n_spikes
    assert not np.all(session.model.spike_clusters[spike_ids] == sc)

    assert 'klustakwik_version' not in session.model.clustering_metadata
    kk_dir = op.join(session.settings.exp_settings_dir, 'klustakwik2')

    # Check temporary files with latest clustering.
    files = os.listdir(kk_dir)
    assert 'spike_clusters.txt' in files
    if len(files) >= 2:
        assert 'spike_clusters.txt~' in files
    sc_txt = np.loadtxt(op.join(kk_dir, 'spike_clusters.txt'))
    ae(sc, sc_txt[spike_ids])

    for clustering in ('test',):
        session.change_clustering(clustering)
        assert len(session.model.spike_clusters) == n_spikes
        assert np.all(session.model.spike_clusters[spike_ids] == sc)

    # Test clustering metadata.
    session.change_clustering('test')
    metadata = session.model.clustering_metadata
    assert 'klustakwik_version' in metadata
    assert metadata['klustakwik_num_starting_clusters'] == 10


def test_session_detect(tempdir):
    channels = range(n_channels)
    graph = [[i, i + 1] for i in range(n_channels - 1)]
    probe = {'channel_groups': {
             0: {'channels': channels,
                 'graph': graph,
                 }}}
    sample_rate = 10000
    n_samples_traces = 10000
    traces = artificial_traces(n_samples_traces, n_channels)
    assert traces is not None

    kwik_path = op.join(tempdir, 'test.kwik')
    create_kwik(kwik_path=kwik_path, probe=probe, sample_rate=sample_rate)
    session = Session(kwik_path, phy_user_dir=tempdir)
    session.detect(traces=traces)
    m = session.model
    if m.n_spikes > 0:
        shape = (m.n_spikes, n_channels * m.n_features_per_channel)
        assert m.features.shape == shape
