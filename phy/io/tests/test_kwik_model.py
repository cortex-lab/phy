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
                              artificial_spike_clusters,
                              artificial_features,
                              artificial_masks)
from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5
from ..kwik_model import (KwikModel, _list_channel_groups, _list_recordings,
                          _list_clusterings, _kwik_filenames)


#------------------------------------------------------------------------------
# Utility test routines
#------------------------------------------------------------------------------

def _create_test_file(dir_path, n_clusters=None, n_spikes=None,
                      n_channels=None):
    """Create a test kwik file."""
    filename = op.join(dir_path, '_test.kwik')
    filenames = _kwik_filenames(filename)
    kwx_filename = filenames['kwx']

    # Create the kwik file.
    with open_h5(filename, 'w') as f:
        f.write_attr('/', 'kwik_version', 2)
        f.write_attr('/application_data/spikedetekt',
                     'nfeatures_per_channel', 2)

        # Create spike times.
        spike_times = artificial_spike_times(n_spikes).astype(np.int64)
        f.write('/channel_groups/1/spikes/time_samples', spike_times)

        # Create spike clusters.
        spike_clusters = artificial_spike_clusters(n_spikes,
                                                   n_clusters).astype(np.int32)
        f.write('/channel_groups/1/spikes/clusters/main', spike_clusters)

        # Create cluster metadata.
        for cluster in range(n_clusters):
            group = '/channel_groups/1/clusters/main/{0:d}'.format(cluster)
            color = ('/channel_groups/1/clusters/main/{0:d}'.format(cluster) +
                     '/application_data/klustaviewa')
            f.write_attr(group, 'cluster_group', 3)
            f.write_attr(color, 'color', randint(2, 10))

        # Create recordings.
        f.write_attr('/recordings/0', 'name', 'recording_0')

    # Create the kwx file.
    with open_h5(kwx_filename, 'w') as f:
        f.write_attr('/', 'kwik_version', 2)
        features = artificial_features(n_spikes, n_channels * 3)
        masks = artificial_masks(n_spikes, n_channels * 3)
        fm = np.dstack((features, masks)).astype(np.float32)
        f.write('/channel_groups/1/features_masks', fm)

    return filename


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_kwik_utility():

    n_clusters = 10
    n_spikes = 1000
    n_channels = 28
    # n_features_per_channel = 2

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels)
        model = KwikModel(filename)

        assert _list_channel_groups(model._kwik.h5py_file) == [1]
        assert _list_recordings(model._kwik.h5py_file) == [0]
        assert _list_clusterings(model._kwik.h5py_file, 1) == ['main']


def test_kwik_open():

    n_clusters = 10
    n_spikes = 1000
    n_channels = 28
    # n_features_per_channel = 2

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels)

        # Test implicit open() method.
        kwik = KwikModel(filename)
        assert kwik.spike_times[:].shape == (n_spikes,)
        assert kwik.spike_clusters[:].shape == (n_spikes,)
        assert kwik.spike_clusters[:].min() == 0
        assert kwik.spike_clusters[:].max() == n_clusters - 1
