# -*- coding: utf-8 -*-

"""Tests of mock Kwik file creation."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from random import randint

import numpy as np
from numpy.testing import assert_array_equal as ae
import h5py
from pytest import raises

from ..artificial import (artificial_spike_times,
                          artificial_spike_clusters,
                          artificial_features,
                          artificial_masks,
                          artificial_traces)
from ....electrode.mea import MEA, staggered_positions
from ....utils.tempdir import TemporaryDirectory
from ...h5 import open_h5
from ...kwik_model import (KwikModel, _list_channel_groups, _list_channels,
                           _list_recordings,
                           _list_clusterings, _kwik_filenames)
from ..kwik import create_mock_kwik


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_create_kwik():

    n_clusters = 10
    n_spikes = 1000
    n_channels = 28
    n_fets = 2
    n_samples_traces = 2000

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
