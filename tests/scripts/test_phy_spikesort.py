# -*- coding: utf-8 -*-1

"""Tests of phy spike sorting commands."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from subprocess import call

from ...io.kwik.mock import create_mock_kwik


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _call(cmd):
    ret = call(cmd.split(' '))
    if ret != 0:
        raise RuntimeError()


n_clusters = 5
n_spikes = 50
n_channels = 28
n_fets = 2
n_samples_traces = 3000


def test_script_run(tempdir):

    # Create the test HDF5 file in the temporary directory.
    kwik_path = create_mock_kwik(tempdir,
                                 n_clusters=n_clusters,
                                 n_spikes=n_spikes,
                                 n_channels=n_channels,
                                 n_features_per_channel=n_fets,
                                 n_samples_traces=n_samples_traces,
                                 add_original=False,
                                 )

    _call('phy -v')
    _call('phy -h')
    _call('phy describe ' + kwik_path)

    cmd = ('phy cluster-auto {} --clustering=original')
    _call(cmd.format(kwik_path))

    _call('phy describe {} --clustering=original'.format(kwik_path))
