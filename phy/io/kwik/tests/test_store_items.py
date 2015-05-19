# -*- coding: utf-8 -*-

"""Tests of Kwik store items."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ....electrode.mea import MEA, staggered_positions
from ....utils.tempdir import TemporaryDirectory
from ..model import (KwikModel,
                     _list_channel_groups,
                     _list_channels,
                     _list_recordings,
                     _list_clusterings,
                     _concatenate_spikes,
                     )
from ..mock import create_mock_kwik


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

        # We initialize the ClusterStore.
        cs = ClusterStore(model=model,
                          path=tempdir,
                          spikes_per_cluster=spikes_per_cluster,
                          )

        # We create a n_spikes item to be stored in memory,
        # and we define how to generate it for a given cluster.
        class MyItem(StoreItem):
            name = 'my item'
            fields = [('n_spikes', 'memory')]

            def store_cluster(self, cluster, spikes=None, mode=None):
                self.memory_store.store(cluster, n_spikes=len(spikes))

            def on_cluster(self, up):
                if up.description == 'merge':
                    n = sum(len(up.old_spikes_per_cluster[cl])
                            for cl in up.deleted)
                    self.memory_store.store(up.added[0], n_spikes=n)
                else:
                    super(MyItem, self).on_cluster(up)

        cs.register_item(MyItem)

        # Now we generate the store.
        cs.generate()


