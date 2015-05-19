# -*- coding: utf-8 -*-

"""Tests of Kwik store items."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ....utils.array import _spikes_per_cluster
from ....utils.tempdir import TemporaryDirectory
from ..model import (KwikModel,
                     )
from ..mock import create_mock_kwik
from ...store import ClusterStore
from ..store_items import FeatureMasks, Waveforms, ClusterStatistics


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

_N_CLUSTERS = 15
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

        model = KwikModel(filename)
        spc = _spikes_per_cluster(np.arange(_N_SPIKES), model.spike_clusters)

        # We initialize the ClusterStore.
        cs = ClusterStore(model=model,
                          path=tempdir,
                          spikes_per_cluster=spc,
                          )

        cs.register_item(FeatureMasks, chunk_size=15)

        # Now we generate the store.
        cs.generate()
