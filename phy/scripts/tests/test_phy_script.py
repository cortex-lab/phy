# -*- coding: utf-8 -*-1

"""Tests of the script."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from subprocess import call

from pytest import mark

from ...utils.tempdir import TemporaryDirectory
from ...io.kwik.mock import create_mock_kwik
from ..phy_script import ParserCreator


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


def test_script_parser():

    p = ParserCreator()

    kwik_path = 'test'

    args = p.parse(['cluster-manual', kwik_path, '-i', '--debug'])
    assert args.command == 'cluster-manual'
    assert args.ipython
    assert args.debug
    assert not args.profiler
    assert not args.line_profiler

    args = p.parse(['cluster-auto', kwik_path, '-lp'])
    assert args.command == 'cluster-auto'
    assert not args.ipython
    assert not args.debug
    assert not args.profiler
    assert args.line_profiler


@mark.long
def test_script_run():

    with TemporaryDirectory() as tmpdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tmpdir,
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
