# -*- coding: utf-8 -*-

"""Multi-electrode arrays."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import itertools

import numpy as np

from ..utils._types import _as_array


#------------------------------------------------------------------------------
# PRB file utilities
#------------------------------------------------------------------------------

def _edges_to_adjacency_list(edges):
    """Convert a list of edges into an adjacency list."""
    adj = {}
    for i, j in edges:
        if i in adj:
            ni = adj[i]
        else:
            ni = adj[i] = set()
        if j in adj:
            nj = adj[j]
        else:
            nj = adj[j] = set()
        ni.add(j)
        nj.add(i)
    return adj


def _probe_positions(probe, group):
    """Return the positions of a probe channel group."""
    positions = probe['channel_groups'][group]['geometry']
    channels = _probe_channels(probe, group)
    return np.array([positions[channel] for channel in channels])


def _probe_channels(probe, group):
    """Return the list of channels in a channel group.

    The order is kept.

    """
    return probe['channel_groups'][group]['channels']


def _probe_all_channels(probe):
    """Return the list of channels in the probe."""
    cgs = probe['channel_groups'].values()
    cg_channels = [cg['channels'] for cg in cgs]
    return sorted(set(itertools.chain(*cg_channels)))


def _probe_adjacency_list(probe):
    """Return an adjacency list of a whole probe."""
    cgs = probe['channel_groups'].values()
    graphs = [cg['graph'] for cg in cgs]
    edges = list(itertools.chain(*graphs))
    adjacency_list = _edges_to_adjacency_list(edges)
    return adjacency_list


def _channels_per_group(probe):
    groups = probe['channel_groups'].keys()
    return {group: probe['channel_groups'][group]['channels']
            for group in groups}


#------------------------------------------------------------------------------
# MEA class
#------------------------------------------------------------------------------

class MEA(object):
    """A Multi-Electrode Array.

    There are two modes:

    * No probe specified: one single channel group, positions and adjacency
      list specified directly.
    * Probe specified: one can change the current channel_group.

    """

    def __init__(self, channels=None,
                 positions=None,
                 adjacency=None,
                 probe=None,
                 ):
        self._probe = probe
        self._channels = channels
        if positions is not None:
            assert self.n_channels == positions.shape[0]
        self._positions = positions
        # This is a mapping {channel: list of neighbors}.
        if adjacency is None and probe is not None:
            adjacency = _probe_adjacency_list(probe)
            self.channels_per_group = _channels_per_group(probe)
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
        return len(self._channels) if self._channels is not None else 0

    @property
    def adjacency(self):
        """Adjacency graph."""
        return self._adjacency

    @adjacency.setter
    def adjacency(self, value):
        self._adjacency = value

    def change_channel_group(self, group):
        assert self._probe is not None
        self._channels = _probe_channels(self._probe, group)
        self._positions = _probe_positions(self._probe, group)


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
