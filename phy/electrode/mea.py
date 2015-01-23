# -*- coding: utf-8 -*-

"""Multi-electrode arrays."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# MEA facilities
#------------------------------------------------------------------------------

def normalize_positions(positions):
    """Normalize channel positions into [-1, 1]."""
    # TODO: add 'keep_ratio' option.
    min, max = positions.min(), positions.max()
    positions_n = (positions - min) / float(max - min)
    positions_n = -1. + 2. * positions_n
    return positions_n


class MEA(object):
    def __init__(self, positions=None, adjacency=None, n_channels=None):
        if positions is not None and n_channels is None:
            n_channels = positions.shape[0]
        self._n_channels = n_channels
        self.positions = positions
        self.adjacency = adjacency

    def _check_positions(self, positions):
        if positions is not None:
            positions = np.asarray(positions)
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
