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

_N_CLUSTERS = 10
_N_SPIKES = 50
_N_CHANNELS = 28
_N_FETS = 2
_N_SAMPLES_TRACES = 3000


def test_kwik_utility():

    channels = list(range(_N_CHANNELS))

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=_N_CLUSTERS,
                                    n_spikes=_N_SPIKES,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES)
        model = KwikModel(filename)

        assert _list_channel_groups(model._kwik.h5py_file) == [1]
        assert _list_recordings(model._kwik.h5py_file) == [0]
        assert _list_clusterings(model._kwik.h5py_file, 1) == ['main']
        assert _list_channels(model._kwik.h5py_file, 1) == channels


def test_kwik_open():

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=_N_CLUSTERS,
                                    n_spikes=_N_SPIKES,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES)

        with raises(ValueError):
            KwikModel()

        # Test implicit open() method.
        kwik = KwikModel(filename)

        kwik.metadata
        assert kwik.channels == list(range(_N_CHANNELS))
        assert kwik.n_channels == _N_CHANNELS
        assert kwik.n_spikes == _N_SPIKES

        assert kwik.spike_times[:].shape == (_N_SPIKES,)

        assert kwik.spike_clusters[:].shape == (_N_SPIKES,)
        assert kwik.spike_clusters[:].min() == 0
        assert kwik.spike_clusters[:].max() == _N_CLUSTERS - 1

        assert kwik.features.shape == (_N_SPIKES,
                                       _N_CHANNELS * _N_FETS)
        kwik.features[0, ...]

        assert kwik.masks.shape == (_N_SPIKES, _N_CHANNELS)

        assert kwik.traces.shape == (_N_SAMPLES_TRACES, _N_CHANNELS)

        # TODO: fix this
        # print(kwik.waveforms[0].shape)
        assert kwik.waveforms[10].shape == (1, 40, _N_CHANNELS)
        assert kwik.waveforms[[10, 20]].shape == (2, 40, _N_CHANNELS)

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
        assert kwik.probe.positions.shape == (_N_CHANNELS, 2)
        ae(kwik.probe.positions, staggered_positions(_N_CHANNELS))

        # Not implemented yet.
        with raises(NotImplementedError):
            kwik.save()

        kwik.close()


def test_kwik_open_no_kwx():

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=_N_CLUSTERS,
                                    n_spikes=_N_SPIKES,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES,
                                    with_kwx=False)

        # Test implicit open() method.
        kwik = KwikModel(filename)
        kwik.close()


def test_kwik_open_no_kwd():

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=_N_CLUSTERS,
                                    n_spikes=_N_SPIKES,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES,
                                    with_kwd=False)

        # Test implicit open() method.
        kwik = KwikModel(filename)
        kwik.close()
