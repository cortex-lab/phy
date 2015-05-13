# -*- coding: utf-8 -*-

"""Tests of mock Kwik file creation."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ....utils.tempdir import TemporaryDirectory
from ...h5 import open_h5
from ..mock import create_mock_kwik


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_create_kwik():

    n_clusters = 10
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

        with open_h5(filename) as f:
            assert f
