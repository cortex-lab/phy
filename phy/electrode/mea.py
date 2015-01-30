# -*- coding: utf-8 -*-

"""Multi-electrode arrays."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils.array import _as_array


#------------------------------------------------------------------------------
# MEA facilities
#------------------------------------------------------------------------------

class MEA(object):
    def __init__(self, positions=None, adjacency=None, n_channels=None):
        if positions is not None and n_channels is None:
            n_channels = positions.shape[0]
        self._n_channels = n_channels
        self.positions = positions
        self.adjacency = adjacency

    def _check_positions(self, positions):
        if positions is not None:
            positions = _as_array(positions)
            if self._n_channels is None:
                self._n_channels = positions.shape[0]
            if positions.shape[0] != self._n_channels:
                raise ValueError("'positions' "
                                 "(shape {0:s})".format(str(positions.shape)) +
                                 " and 'n_channels' "
                                 "({0:d})".format(self.n_channels) +
                                 " do not match.")

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._check_positions(value)
        self._positions = value

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, value):
        self._adjacency = value


#------------------------------------------------------------------------------
# Common probes
#------------------------------------------------------------------------------

def staggered_positions(n_channels):
    """Generate channel positions for a staggered probe."""
    i = np.arange(n_channels - 1)
    x, y = (-1) ** i * (5 + i), 10 * (i + 1)
    return np.flipud(np.r_[np.zeros((1, 2)), np.c_[x, y]])
