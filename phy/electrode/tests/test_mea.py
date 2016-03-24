# -*- coding: utf-8 -*-

"""Test MEA."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import raises
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..mea import (_probe_channels, _remap_adjacency, _adjacency_subset,
                   _probe_positions, _probe_adjacency_list,
                   MEA, linear_positions, staggered_positions,
                   load_probe, list_probes
                   )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_remap():
    adjacency = {1: [2, 3, 7], 3: [5, 11]}
    mapping = {1: 3, 2: 20, 3: 30, 5: 50, 7: 70, 11: 1}
    remapped = _remap_adjacency(adjacency, mapping)
    assert sorted(remapped.keys()) == [3, 30]
    assert remapped[3] == [20, 30, 70]
    assert remapped[30] == [50, 1]


def test_adjacency_subset():
    adjacency = {1: [2, 3, 7], 3: [5, 11], 5: [1, 2, 11]}
    subset = [1, 5, 32]
    adjsub = _adjacency_subset(adjacency, subset)
    assert sorted(adjsub.keys()) == [1, 5]
    assert adjsub[1] == []
    assert adjsub[5] == [1]


def test_probe():
    probe = {'channel_groups': {
             0: {'channels': [0, 3, 1],
                 'graph': [[0, 3], [1, 0]],
                 'geometry': {0: (10, 10), 1: (10, 20), 3: (20, 30)},
                 },
             1: {'channels': [7],
                 'graph': [],
                 },
             }}
    adjacency = {0: set([1, 3]),
                 1: set([0]),
                 3: set([0]),
                 }
    assert _probe_channels(probe, 0) == [0, 3, 1]
    ae(_probe_positions(probe, 0), [(10, 10), (20, 30), (10, 20)])
    assert _probe_adjacency_list(probe) == adjacency

    mea = MEA(probe=probe)

    assert mea.adjacency == adjacency
    assert mea.channels_per_group == {0: [0, 3, 1], 1: [7]}
    assert mea.channels == [0, 3, 1]
    assert mea.n_channels == 3
    ae(mea.positions, [(10, 10), (20, 30), (10, 20)])


def test_mea():

    n_channels = 10
    channels = np.arange(n_channels)
    positions = np.random.randn(n_channels, 2)

    mea = MEA(channels, positions=positions)
    ae(mea.positions, positions)
    assert mea.adjacency is None

    mea = MEA(channels, positions=positions)
    assert mea.n_channels == n_channels

    mea = MEA(channels, positions=positions)
    assert mea.n_channels == n_channels

    with raises(ValueError):
        MEA(channels=np.arange(n_channels + 1), positions=positions)

    with raises(ValueError):
        MEA(channels=channels, positions=positions[:-1, :])


def test_positions():
    probe = staggered_positions(31)
    assert probe.shape == (31, 2)
    ae(probe[-1], (0, 0))

    probe = linear_positions(29)
    assert probe.shape == (29, 2)


def test_library(tempdir):
    assert '1x32_buzsaki' in list_probes()

    probe = load_probe('1x32_buzsaki')
    assert probe
    assert probe.channels == list(range(32))

    path = op.join(tempdir, 'test.prb')
    with raises(IOError):
        load_probe(path)

    with open(path, 'w') as f:
        f.write('')
    with raises(KeyError):
        load_probe(path)
