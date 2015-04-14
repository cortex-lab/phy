# -*- coding: utf-8 -*-

"""Test MEA."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..mea import MEA, linear_positions, staggered_positions


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_mea():

    n_channels = 10
    channels = np.arange(n_channels)
    positions = np.random.randn(n_channels, 2)

    mea = MEA(channels)
    mea.positions = positions
    ae(mea.positions, positions)
    assert mea.adjacency is None

    mea = MEA(channels, positions=positions)
    assert mea.n_channels == n_channels

    mea = MEA(channels, positions=positions)
    assert mea.n_channels == n_channels

    with raises(AssertionError):
        MEA(channels=np.arange(n_channels + 1), positions=positions)

    with raises(AssertionError):
        MEA(channels=channels, positions=positions[:-1, :])

    mea = MEA(channels=channels)
    assert mea.n_channels == n_channels
    mea.positions = positions
    with raises(ValueError):
        mea.positions = positions[:-1, :]


def test_positions():
    probe = staggered_positions(31)
    assert probe.shape == (31, 2)
    ae(probe[-1], (0, 0))

    probe = linear_positions(29)
    assert probe.shape == (29, 2)
