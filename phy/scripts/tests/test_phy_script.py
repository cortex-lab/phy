# -*- coding: utf-8 -*-1

"""Tests of the script."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from subprocess import call
import time

import numpy as np
from pytest import raises, mark

from ...utils.tempdir import TemporaryDirectory
from ...io.kwik.mock import create_mock_kwik
from ...cluster.session import Session
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


def _parse(args):
    return _parse_args(args)[0]


def test_script_parser():

    p = ParserCreator()

    kwik_path = 'test'

    args, _ = p.parse(['-i', '--debug', 'cluster-manual', kwik_path])
    assert args.command == 'cluster-manual'
    assert args.ipython
    assert args.debug
    assert not args.profiler
    assert not args.line_profiler

    args, _ = p.parse(['-lp', 'cluster-auto', kwik_path])
    assert args.command == 'cluster-auto'
    assert not args.ipython
    assert not args.debug
    assert not args.profiler
    assert args.line_profiler

    # Test extra arguments.
    args, kwargs = p.parse(['cluster-auto',
                            kwik_path,
                            '--aa=foo',
                            '--b-b=1.5',
                            '--c_c=2',
                            ])
    assert kwargs['aa'] == 'foo'
    assert kwargs['b_b'] == 1.5
    assert kwargs['c_c'] == 2


@mark.long
def test_script_run():

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

        _call('phy -v')
        _call('phy -h')
        _call('phy describe ' + kwik_path)

        cmd = ('phy cluster-auto {} --num_starting_clusters=10 '
               '--clustering=auto')
        _call(cmd.format(kwik_path))
        time.sleep(.5)

        session = Session(kwik_path)
        session.change_clustering('auto')
        assert np.all(session.model.spike_clusters == 0)
