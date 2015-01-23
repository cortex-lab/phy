# -*- coding: utf-8 -*-

"""Multi-electrode arrays."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# MEA facilities
#------------------------------------------------------------------------------

class MEA(object):
    def __init__(self, positions=None, adjacency=None, n_channels=None):
        # if n_channels is not None and positions is not None:
        #     assert n_channels == positions.shape[0]
        if positions is not None:
            n_channels = positions.shape[0]
        self._n_channels = n_channels
        self.positions = positions
        self.adjacency = adjacency

    def _check_positions(self, positions):
        if positions is not None:
            assert isinstance(positions, np.ndarray)
            assert positions.shape[0] == self._n_channels

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
