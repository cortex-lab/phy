# -*- coding: utf-8 -*-

"""Tests of Kwik file opening routines."""

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

from ...io.mock.artificial import (artificial_spike_times,
                                   artificial_spike_clusters,
                                   artificial_features,
                                   artificial_masks,
                                   artificial_traces)
from ...electrode.mea import MEA, staggered_positions
from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5
from ..kwik_model import (KwikModel, _list_channel_groups, _list_channels,
                          _list_recordings,
                          _list_clusterings, _kwik_filenames)
from ..mock.kwik import create_mock_kwik


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
        filename = create_mock_kwik(tempdir,
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
        filename = create_mock_kwik(tempdir,
                                    n_clusters=n_clusters,
                                    n_spikes=n_spikes,
                                    n_channels=n_channels,
                                    n_features_per_channel=n_fets,
                                    n_samples_traces=n_samples_traces)

        with raises(ValueError):
            KwikModel()

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
        kwik.features[0, ...]

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

        # TODO: test cluster_metadata.
        kwik.cluster_metadata

        # Test probe.
        assert isinstance(kwik.probe, MEA)
        assert kwik.probe.positions.shape == (n_channels, 2)
        ae(kwik.probe.positions, staggered_positions(n_channels))

        # Not implemented yet.
        with raises(NotImplementedError):
            kwik.save()

        kwik.close()


def test_kwik_open_no_kwx():

    n_clusters = 8
    n_spikes = 100
    n_channels = 4
    n_fets = 1
    n_samples_traces = 1000

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=n_clusters,
                                    n_spikes=n_spikes,
                                    n_channels=n_channels,
                                    n_features_per_channel=n_fets,
                                    n_samples_traces=n_samples_traces,
                                    with_kwx=False)

        # Test implicit open() method.
        kwik = KwikModel(filename)
        kwik.close()


def test_kwik_open_no_kwd():

    n_clusters = 8
    n_spikes = 100
    n_channels = 4
    n_fets = 1
    n_samples_traces = 1000

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=n_clusters,
                                    n_spikes=n_spikes,
                                    n_channels=n_channels,
                                    n_features_per_channel=n_fets,
                                    n_samples_traces=n_samples_traces,
                                    with_kwd=False)

        # Test implicit open() method.
        kwik = KwikModel(filename)
        kwik.close()
