# -*- coding: utf-8 -*-

"""Tests of Kwik file creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import raises

from ...h5 import open_h5
from ..creator import KwikCreator, _write_by_chunk, create_kwik
from ..mock import (artificial_spike_samples,
                    artificial_features,
                    artificial_masks,
                    )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_write_by_chunk(tempdir):
    n = 5
    arrs = [np.random.rand(i + 1, 3).astype(np.float32) for i in range(n)]
    n_tot = sum(_.shape[0] for _ in arrs)

    path = op.join(tempdir, 'test.h5')
    with open_h5(path, 'w') as f:
        ds = f.write('/test', shape=(n_tot, 3), dtype=np.float32)
        _write_by_chunk(ds, arrs)
    with open_h5(path, 'r') as f:
        ds = f.read('/test')[...]
        offset = 0
        for i, arr in enumerate(arrs):
            size = arr.shape[0]
            assert size == (i + 1)
            ac(ds[offset:offset + size, ...], arr)
            offset += size


def test_creator_simple(tempdir):
    basename = op.join(tempdir, 'my_file')

    creator = KwikCreator(basename)

    # Test create empty files.
    creator.create_empty()
    assert op.exists(basename + '.kwik')
    assert op.exists(basename + '.kwx')

    # Test metadata.
    creator.set_metadata('/application_data/spikedetekt',
                         a=1, b=2., c=[0, 1])

    with open_h5(creator.kwik_path, 'r') as f:
        assert f.read_attr('/application_data/spikedetekt', 'a') == 1
        assert f.read_attr('/application_data/spikedetekt', 'b') == 2.
        ae(f.read_attr('/application_data/spikedetekt', 'c'), [0, 1])

    # Test add spikes in one block.
    n_spikes = 100
    n_channels = 8
    n_features = 3

    spike_samples = artificial_spike_samples(n_spikes)
    features = artificial_features(n_spikes, n_channels, n_features)
    masks = artificial_masks(n_spikes, n_channels)

    creator.add_spikes(group=0,
                       spike_samples=spike_samples,
                       features=features.astype(np.float32),
                       masks=masks.astype(np.float32),
                       n_channels=n_channels,
                       n_features=n_features,
                       )

    # Test the spike samples.
    with open_h5(creator.kwik_path, 'r') as f:
        s = f.read('/channel_groups/0/spikes/time_samples')[...]
        assert s.dtype == np.uint64
        ac(s, spike_samples)

    # Test the features and masks.
    with open_h5(creator.kwx_path, 'r') as f:
        fm = f.read('/channel_groups/0/features_masks')[...]
        assert fm.dtype == np.float32
        ac(fm[:, :, 0], features.reshape((-1, n_channels * n_features)))
        ac(fm[:, ::n_features, 1], masks)

    # Spikes can only been added once.
    with raises(RuntimeError):
        creator.add_spikes(group=0,
                           spike_samples=spike_samples,
                           n_channels=n_channels,
                           n_features=n_features)


def test_creator_chunks(tempdir):
    basename = op.join(tempdir, 'my_file')

    creator = KwikCreator(basename)
    creator.create_empty()

    # Test add spikes in one block.
    n_spikes = 100
    n_channels = 8
    n_features = 3

    spike_samples = artificial_spike_samples(n_spikes)
    features = artificial_features(n_spikes, n_channels,
                                   n_features).astype(np.float32)
    masks = artificial_masks(n_spikes, n_channels).astype(np.float32)

    def _split(arr):
        n = n_spikes // 10
        return [arr[k:k + n, ...] for k in range(0, n_spikes, n)]

    creator.add_spikes(group=0,
                       spike_samples=spike_samples,
                       features=_split(features),
                       masks=_split(masks),
                       n_channels=n_channels,
                       n_features=n_features,
                       )

    # Test the spike samples.
    with open_h5(creator.kwik_path, 'r') as f:
        s = f.read('/channel_groups/0/spikes/time_samples')[...]
        assert s.dtype == np.uint64
        ac(s, spike_samples)

    # Test the features and masks.
    with open_h5(creator.kwx_path, 'r') as f:
        fm = f.read('/channel_groups/0/features_masks')[...]
        assert fm.dtype == np.float32
        ac(fm[:, :, 0], features.reshape((-1, n_channels * n_features)))
        ac(fm[:, ::n_features, 1], masks)


def test_creator_add_no_spikes(tempdir):
    basename = op.join(tempdir, 'my_file')

    creator = KwikCreator(basename)
    creator.create_empty()

    creator.add_spikes(group=0, n_channels=4, n_features=2)


def test_creator_metadata(tempdir):
    basename = op.join(tempdir, 'my_file')

    creator = KwikCreator(basename)

    # Add recording.
    creator.add_recording(0, start_sample=20000, sample_rate=10000)

    with open_h5(creator.kwik_path, 'r') as f:
        assert f.read_attr('/recordings/0', 'start_sample') == 20000
        assert f.read_attr('/recordings/0', 'start_time') == 2.
        assert f.read_attr('/recordings/0', 'sample_rate') == 10000

    # Add probe.
    channels = [0, 3, 1]
    graph = [[0, 3], [1, 0]]
    probe = {'channel_groups': {
             0: {'channels': channels,
                 'graph': graph,
                 'geometry': {0: (10, 10)},
                 }}}
    creator.set_probe(probe)

    with open_h5(creator.kwik_path, 'r') as f:
        ae(f.read_attr('/channel_groups/0', 'channel_order'), channels)
        ae(f.read_attr('/channel_groups/0', 'adjacency_graph'), graph)
        for channel in channels:
            path = '/channel_groups/0/channels/{:d}'.format(channel)
            position = (10, 10) if channel == 0 else (0, channel)
            ae(f.read_attr(path, 'position'), position)

    # Add clustering.
    creator.add_clustering(group=0,
                           name='main',
                           spike_clusters=np.arange(10),
                           cluster_groups={3: 1},
                           )

    with open_h5(creator.kwik_path, 'r') as f:
        sc = f.read('/channel_groups/0/spikes/clusters/main')
        ae(sc, np.arange(10))

        for cluster in range(10):
            path = '/channel_groups/0/clusters/main/{:d}'.format(cluster)
            cg = f.read_attr(path, 'cluster_group')
            assert cg == 3 if cluster != 3 else 1


def test_create_kwik(tempdir):

    channels = [0, 3, 1]
    graph = [[0, 3], [1, 0]]
    probe = {'channel_groups': {
             0: {'channels': channels,
                 'graph': graph,
                 'geometry': {0: (10, 10)},
                 }}}

    kwik_path = op.join(tempdir, 'test.kwik')
    create_kwik(kwik_path=kwik_path,
                probe=probe,
                sample_rate=20000,
                )
