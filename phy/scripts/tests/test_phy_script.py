# -*- coding: utf-8 -*-1

"""Tests of the script."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from subprocess import call

import numpy as np
from pytest import raises

from ...utils.tempdir import TemporaryDirectory
from ...io.kwik.mock import create_mock_kwik
from ...cluster.session import Session
from ..phy_script import _parse_args


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

    kwik_path = 'test'

    for cmd in ('cluster-manual', 'cluster-auto'):
        assert _parse([cmd, kwik_path]).command == cmd

    args = _parse(['cluster-manual', '-i', '--debug', kwik_path])
    assert args.command == 'cluster-manual'
    assert args.ipython
    assert args.debug
    assert not args.profiler
    assert not args.line_profiler

    args = _parse(['cluster-auto', '-lp', kwik_path])
    assert args.command == 'cluster-auto'
    assert not args.ipython
    assert not args.debug
    assert not args.profiler
    assert args.line_profiler

    # Test extra arguments.
    args, kwargs = _parse_args(['cluster-auto',
                                kwik_path,
                                '--a=foo',
                                '--b-b=1.5',
                                '--c=2',
                                ])
    assert kwargs['a'] == 'foo'
    assert kwargs['b_b'] == 1.5
    assert kwargs['c'] == 2


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

        cmd = ('phy cluster-auto {} --num_starting_clusters=10 '
               '--clustering-name=auto')
        _call(cmd.format(kwik_path))

        session = Session(kwik_path)
        session.change_clustering('auto')
        assert np.all(session.model.spike_clusters == 0)
