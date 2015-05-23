# -*- coding: utf-8 -*-

"""Tests of Kwik store items."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ....utils.array import _spikes_per_cluster
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
        spc = _spikes_per_cluster(np.arange(_N_SPIKES), model.spike_clusters)
        clusters = sorted(spc.keys())

        # We initialize the ClusterStore.
        cs = create_store(model,
                          path=tempdir,
                          spikes_per_cluster=spc,
                          features_masks_chunk_size=15,
                          waveforms_n_spikes_max=5,
                          waveforms_excerpt_size=2,
                          )

        # We add a custom statistic function.
        def mean_features_bis(cluster):
            fet = cs.features(cluster)
            cs.memory_store.store(cluster, mean_features_bis=fet.mean(axis=0))

        cs.items['statistics'].add('mean_features_bis', mean_features_bis)
        cs.register_field('mean_features_bis', 'statistics')

        # Now we generate the store.
        cs.generate()

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
            item = cs.items['waveforms']
            waveforms_spikes = item.load_waveforms_spikes(cluster)
            waveforms_expected = model.waveforms[waveforms_spikes]
            ae(waveforms_store, waveforms_expected)

            # Check some statistics.
            ae(cs.mean_masks(cluster),
               masks_expected.mean(axis=0))
            ae(cs.mean_waveforms(cluster),
               waveforms_expected.mean(axis=0))
            ae(cs.mean_features_bis(cluster),
               fet_expected.mean(axis=0))
