# -*- coding: utf-8 -*-

"""Test MEA."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises
import numpy as np

from ..mea import MEA


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_mea():
    mea = MEA()

    n_channels = 10
    positions = np.random.randn(n_channels, 2)

    mea = MEA(positions=positions)
    assert mea.n_channels == n_channels

    mea = MEA(positions=positions, n_channels=n_channels)
    assert mea.n_channels == n_channels

    with raises(ValueError):
        MEA(positions=positions, n_channels=n_channels+1)

    with raises(ValueError):
        MEA(positions=positions[:-1, :], n_channels=n_channels)

    mea = MEA(n_channels=n_channels)
    mea.positions = positions
    with raises(ValueError):
        mea.positions = positions[:-1, :]
