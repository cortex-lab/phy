# -*- coding: utf-8 -*-

"""Multi-electrode arrays."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils._types import _as_array


#------------------------------------------------------------------------------
# MEA facilities
#------------------------------------------------------------------------------

class MEA(object):
    """A Multi-Electrode Array."""

    # TODO:
    # * multi-shank

    def __init__(self, channels, positions=None, adjacency=None):
        # channels is a list of unique channel identifiers, not necessarily
        # in increasing order.
        # positions must have the same number of rows than the number of
        # channels.
        self._channels = channels
        if positions is not None:
            assert self.n_channels == positions.shape[0]
        self._positions = positions
        self._adjacency = adjacency

    def _check_positions(self, positions):
        if positions is None:
            return
        positions = _as_array(positions)
        if self.n_channels is None:
            self.n_channels = positions.shape[0]
        if positions.shape[0] != self.n_channels:
            raise ValueError("'positions' "
                             "(shape {0:s})".format(str(positions.shape)) +
                             " and 'n_channels' "
                             "({0:d})".format(self.n_channels) +
                             " do not match.")

    @property
    def positions(self):
        """Channel positions."""
        return self._positions

    @positions.setter
    def positions(self, value):
        self._check_positions(value)
        self._positions = value

    @property
    def channels(self):
        """Channel ids."""
        return self._channels

    @property
    def n_channels(self):
        """Number of channels."""
        return len(self._channels)

    @property
    def adjacency(self):
        """Adjacency graph."""
        return self._adjacency

    @adjacency.setter
    def adjacency(self, value):
        self._adjacency = value


#------------------------------------------------------------------------------
# Common probes
#------------------------------------------------------------------------------

def linear_positions(n_channels):
    """Linear channel positions along the vertical axis."""
    return np.c_[np.zeros(n_channels),
                 np.linspace(0., 1., n_channels)]


def staggered_positions(n_channels):
    """Generate channel positions for a staggered probe."""
    i = np.arange(n_channels - 1)
    x, y = (-1) ** i * (5 + i), 10 * (i + 1)
    pos = np.flipud(np.r_[np.zeros((1, 2)), np.c_[x, y]])
    return pos
