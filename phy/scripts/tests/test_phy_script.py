# -*- coding: utf-8 -*-1

"""Tests of the script."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from subprocess import call

from pytest import raises

from ...utils.tempdir import TemporaryDirectory
from ...io.kwik.mock import create_mock_kwik


#------------------------------------------------------------------------------
# Script tests
#------------------------------------------------------------------------------

n_clusters = 5
n_spikes = 50
n_channels = 28
n_fets = 2
n_samples_traces = 3000


def _call(cmd):
    ret = call(cmd.split(' '))
    if ret != 0:
        raise RuntimeError()


def test_script():

    with TemporaryDirectory() as tmpdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tmpdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        with raises(RuntimeError):
            _call('phy')

        _call('phy -h')
        _call('phy -v')
