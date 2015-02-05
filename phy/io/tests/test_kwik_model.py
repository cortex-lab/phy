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
                              artificial_masks,
                              artificial_traces)
from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5
from ..kwik_model import (KwikModel, _list_channel_groups, _list_channels,
                          _list_recordings,
                          _list_clusterings, _kwik_filenames)


#------------------------------------------------------------------------------
# Utility test routines
#------------------------------------------------------------------------------

def _create_test_file(dir_path, n_clusters=None, n_spikes=None,
                      n_channels=None, n_features_per_channel=None,
                      n_samples_traces=None):
    """Create a test kwik file."""
    filename = op.join(dir_path, '_test.kwik')
    filenames = _kwik_filenames(filename)
    kwx_filename = filenames['kwx']
    kwd_filename = filenames['raw.kwd']

    # Create the kwik file.
    with open_h5(filename, 'w') as f:
        f.write_attr('/', 'kwik_version', 2)

        def _write_metadata(key, value):
            f.write_attr('/application_data/spikedetekt', key, value)

        _write_metadata('nfeatures_per_channel', n_features_per_channel)
        _write_metadata('extract_s_before', 15)
        _write_metadata('extract_s_after', 25)

        # Create spike times.
        spike_times = artificial_spike_times(n_spikes).astype(np.int64)
        f.write('/channel_groups/1/spikes/time_samples', spike_times)

        # Create spike clusters.
        spike_clusters = artificial_spike_clusters(n_spikes,
                                                   n_clusters).astype(np.int32)
        f.write('/channel_groups/1/spikes/clusters/main', spike_clusters)

        # Create channels.
        for channel in range(n_channels):
            group = '/channel_groups/1/channels/{0:d}'.format(channel)
            f.write_attr(group, 'name', str(channel))

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
        features = artificial_features(n_spikes,
                                       n_channels * n_features_per_channel)
        masks = artificial_masks(n_spikes,
                                 n_channels * n_features_per_channel)
        fm = np.dstack((features, masks)).astype(np.float32)
        f.write('/channel_groups/1/features_masks', fm)

    # Create the raw kwd file.
    with open_h5(kwd_filename, 'w') as f:
        f.write_attr('/', 'kwik_version', 2)
        traces = artificial_traces(n_samples_traces, n_channels)
        # TODO: int16 traces
        f.write('/recordings/0/data', traces.astype(np.float32))

    return filename


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_kwik_utility():

    n_clusters = 10
    n_spikes = 1000
    n_channels = 28
    n_fets = 2
    n_samples_traces = 2000

    channels = list(range(n_channels))

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)
        model = KwikModel(filename)

        assert _list_channel_groups(model._kwik.h5py_file) == [1]
        assert _list_recordings(model._kwik.h5py_file) == [0]
        assert _list_clusterings(model._kwik.h5py_file, 1) == ['main']
        assert _list_channels(model._kwik.h5py_file, 1) == channels


def test_kwik_open():

    n_clusters = 10
    n_spikes = 1000
    n_channels = 28
    n_fets = 2
    n_samples_traces = 2000

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        # Test implicit open() method.
        kwik = KwikModel(filename)

        kwik.metadata
        assert kwik.channels == list(range(n_channels))
        assert kwik.n_channels == n_channels
        assert kwik.n_spikes == n_spikes

        assert kwik.spike_times[:].shape == (n_spikes,)

        assert kwik.spike_clusters[:].shape == (n_spikes,)
        assert kwik.spike_clusters[:].min() == 0
        assert kwik.spike_clusters[:].max() == n_clusters - 1

        assert kwik.features.shape == (n_spikes,
                                       n_channels * n_fets)

        assert kwik.masks.shape == (n_spikes, n_channels)

        assert kwik.traces.shape == (n_samples_traces, n_channels)

        # TODO: fix this
        # print(kwik.waveforms[0].shape)
        assert kwik.waveforms[10].shape == (1, 40, n_channels)
        assert kwik.waveforms[[10, 20]].shape == (2, 40, n_channels)

        with raises(ValueError):
            kwik.clustering = 'foo'
        with raises(ValueError):
            kwik.recording = 47
        with raises(ValueError):
            kwik.channel_group = 42

        # Not implemented yet.
        with raises(NotImplementedError):
            kwik.cluster_metadata
        with raises(NotImplementedError):
            kwik.probe
        with raises(NotImplementedError):
            kwik.save()

        kwik.close()
