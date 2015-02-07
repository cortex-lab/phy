# -*- coding: utf-8 -*-

"""Tests of manual clustering interface."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ..interface import start_manual_clustering
from ....utils.tempdir import TemporaryDirectory
from ....io.mock.artificial import MockModel
from ....io.mock.kwik import create_mock_kwik


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_interface_mock():
    session = start_manual_clustering(model=MockModel())
    view = session.show_waveforms()
    session.select([0])
    view_bis = session.show_waveforms()
    session.merge([3, 4])

    view.close()
    view_bis.close()

    session = start_manual_clustering(model=MockModel())
    session.select([1, 2])
    view = session.show_waveforms()
    view.close()


def test_interface_kwik():

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

        session = start_manual_clustering(filename)
        session.select([0])
        session.merge([3, 4])

        view = session.show_waveforms()

        view.close()
