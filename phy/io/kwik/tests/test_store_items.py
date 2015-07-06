# -*- coding: utf-8 -*-

"""Tests of Kwik store items."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.array import _spikes_per_cluster, _spikes_in_clusters
from ....utils.tempdir import TemporaryDirectory
from ..model import (KwikModel,
                     )
from ..mock import create_mock_kwik
from ..store_items import create_store


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

_N_CLUSTERS = 5
_N_SPIKES = 100
_N_CHANNELS = 28
_N_FETS = 2
_N_SAMPLES_TRACES = 10000


def test_kwik_store():

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=_N_CLUSTERS,
                                    n_spikes=_N_SPIKES,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES)

        nc = _N_CHANNELS - 2
        nf = _N_FETS

        model = KwikModel(filename)
        ns = model.n_samples_waveforms
        spc = _spikes_per_cluster(np.arange(_N_SPIKES), model.spike_clusters)
        clusters = sorted(spc.keys())

        # We initialize the ClusterStore.
        cs = create_store(model,
                          path=tempdir,
                          spikes_per_cluster=spc,
                          features_masks_chunk_size=15,
                          waveforms_chunk_size=15,
                          waveforms_n_spikes_max=5,
                          waveforms_excerpt_size=2,
                          )

        # We add a custom statistic function.
        def mean_features_bis(cluster):
            fet = cs.features(cluster)
            cs.memory_store.store(cluster, mean_features_bis=fet.mean(axis=0))

        cs.items['statistics'].add('mean_features_bis',
                                   mean_features_bis,
                                   (-1, nc),
                                   )
        cs.register_field('mean_features_bis', 'statistics')

        waveforms_item = cs.items['waveforms']

        # Now we generate the store.
        cs.generate()

        # One cluster at a time.
        for cluster in clusters:
            # Check features.
            fet_store = cs.features(cluster)
            fet_expected = model.features[spc[cluster]].reshape((-1, nc, nf))
            ae(fet_store, fet_expected)

            # Check masks.
            masks_store = cs.masks(cluster)
            masks_expected = model.masks[spc[cluster]]
            ae(masks_store, masks_expected)

            # Check waveforms.
            waveforms_store = cs.waveforms(cluster)
            # Find the spikes.
            spikes = waveforms_item.spikes_per_cluster[cluster]
            ae(waveforms_item.spikes_per_cluster[cluster], spikes)
            waveforms_expected = model.waveforms[spikes]
            ae(waveforms_store, waveforms_expected)

            # Check some statistics.
            ae(cs.mean_features(cluster), fet_expected.mean(axis=0))
            ae(cs.mean_features_bis(cluster), fet_expected.mean(axis=0))
            ae(cs.mean_masks(cluster), masks_expected.mean(axis=0))
            ae(cs.mean_waveforms(cluster), waveforms_expected.mean(axis=0))

            assert cs.n_unmasked_channels(cluster) >= 0
            assert cs.main_channels(cluster).shape == (nc,)
            assert cs.mean_probe_position(cluster).shape == (2,)

        # Multiple clusters.
        for clusters in (clusters[::2], [clusters[0]], []):
            n_clusters = len(clusters)
            spikes = _spikes_in_clusters(model.spike_clusters, clusters)
            n_spikes = len(spikes)

            # Features.
            if n_clusters:
                fet_expected = model.features[spikes].reshape((n_spikes,
                                                               nc, nf))
            else:
                fet_expected = np.zeros((0, nc, nf), dtype=np.float32)
            ae(cs.load('features', clusters=clusters), fet_expected)
            ae(cs.load('features', spikes=spikes), fet_expected)

            # Masks.
            if n_clusters:
                masks_expected = model.masks[spikes]
            else:
                masks_expected = np.ones((0, nc), dtype=np.float32)
            ae(cs.load('masks', clusters=clusters), masks_expected)
            ae(cs.load('masks', spikes=spikes), masks_expected)

            # Waveforms.
            spc = waveforms_item.spikes_per_cluster
            if n_clusters:
                spikes = _spikes_in_clusters(spc, clusters)
                waveforms_expected = model.waveforms[spikes]
            else:
                spikes = np.array([], dtype=np.int64)
                waveforms_expected = np.zeros((0, ns, nc), dtype=np.float32)
            ae(cs.load('waveforms', clusters=clusters), waveforms_expected)
            ae(cs.load('waveforms', spikes=spikes), waveforms_expected)

            assert (cs.load('mean_features', clusters=clusters).shape ==
                    (n_clusters, nc, nf))
            assert (cs.load('mean_features_bis', clusters=clusters).shape ==
                    (n_clusters, nc, nf) if n_clusters else (0,))
            assert (cs.load('mean_masks', clusters=clusters).shape ==
                    (n_clusters, nc))
            assert (cs.load('mean_waveforms', clusters=clusters).shape ==
                    (n_clusters, ns, nc))

            assert (cs.load('n_unmasked_channels', clusters=clusters).shape ==
                    (n_clusters,))
            assert (cs.load('main_channels', clusters=clusters).shape ==
                    (n_clusters, nc))
            assert (cs.load('mean_probe_position', clusters=clusters).shape ==
                    (n_clusters, 2))

        # Slice spikes.
        spikes = slice(None, None, 3)

        # Features.
        fet_expected = model.features[spikes].reshape((-1, nc, nf))
        ae(cs.load('features', spikes=spikes), fet_expected)

        # Masks.
        masks_expected = model.masks[spikes]
        ae(cs.load('masks', spikes=spikes), masks_expected)

        # Waveforms.
        waveforms_expected = model.waveforms[spikes]
        ae(cs.load('waveforms', spikes=spikes), waveforms_expected)
