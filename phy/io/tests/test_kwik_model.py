# -*- coding: utf-8 -*-

"""Tests of Kwik file opening routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from random import randint

import numpy as np
import h5py
from pytest import raises

from ...datasets.mock import (artificial_spike_times,
                              artificial_spike_clusters)
from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5
from ..kwik_model import KwikModel


#------------------------------------------------------------------------------
# Utility test routines
#------------------------------------------------------------------------------

def _create_test_file(dir_path, n_clusters=None, n_spikes=None):
    """Create a test kwik file."""
    filename = op.join(dir_path, '_test.kwik')
    with open_h5(filename, 'w') as f:
        spike_times = artificial_spike_times(n_spikes)
        spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
        f.write('/channel_groups/1/spikes/time_samples', spike_times)
        f.write('/channel_groups/1/spikes/clusters/main', spike_clusters)
        for cluster in range(n_clusters):
            group = '/channel_groups/1/clusters/main/{0:d}'.format(cluster)
            color = ('/channel_groups/1/clusters/main/{0:d}'.format(cluster) +
                     '/application_data/klustaviewa')
            f.write_attr(group, 'cluster_group', 3)
            f.write_attr(color, 'color', randint(2, 10))
        return f.filename


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_kwik_open():

    n_clusters = 10
    n_spikes = 1000

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file(tempdir, n_clusters=n_clusters,
                                     n_spikes=n_spikes)

        # Test implicit open() method.
        k = KwikModel(filename, channel_group=1, recording=0)

        assert k.recording == 0
        assert k.channel_group == 1
