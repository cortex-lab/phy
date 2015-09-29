# -*- coding: utf-8 -*-

"""Multi-electrode arrays."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import itertools

import numpy as np

from ..utils._types import _as_array
from ..utils._misc import _read_python


#------------------------------------------------------------------------------
# PRB file utilities
#------------------------------------------------------------------------------

def _edges_to_adjacency_list(edges):
    """Convert a list of edges into an adjacency list."""
    adj = {}
    for i, j in edges:
        if i in adj:  # pragma: no cover
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


def _adjacency_subset(adjacency, subset):
    return {c: [v for v in vals if v in subset]
            for (c, vals) in adjacency.items() if c in subset}


def _remap_adjacency(adjacency, mapping):
    remapped = {}
    for key, vals in adjacency.items():
        remapped[mapping[key]] = [mapping[i] for i in vals]
    return remapped


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


def load_probe(name):
    """Load one of the built-in probes."""
    if op.exists(name):
        # The argument can be either a path to a PRB file.
        path = name
    else:
        # Or the name of a built-in probe.
        curdir = op.realpath(op.dirname(__file__))
        path = op.join(curdir, 'probes/{}.prb'.format(name))
    if not op.exists(path):
        raise IOError("The probe `{}` cannot be found.".format(name))
    return MEA(probe=_read_python(path))


def list_probes():
    """Return the list of built-in probes."""
    curdir = op.realpath(op.dirname(__file__))
    return [op.splitext(fn)[0] for fn in os.listdir(op.join(curdir, 'probes'))
            if fn.endswith('.prb')]


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

    def __init__(self,
                 channels=None,
                 positions=None,
                 adjacency=None,
                 probe=None,
                 ):
        self._probe = probe
        self._channels = channels
        self._check_positions(positions)
        self._positions = positions
        # This is a mapping {channel: list of neighbors}.
        if adjacency is None and probe is not None:
            adjacency = _probe_adjacency_list(probe)
            self.channels_per_group = _channels_per_group(probe)
        self._adjacency = adjacency
        if probe:
            # Select the first channel group.
            cg = sorted(self._probe['channel_groups'].keys())[0]
            self.change_channel_group(cg)

    def _check_positions(self, positions):
        if positions is None:
            return
        positions = _as_array(positions)
        if positions.shape[0] != self.n_channels:
            raise ValueError("'positions' "
                             "(shape {0:s})".format(str(positions.shape)) +
                             " and 'n_channels' "
                             "({0:d})".format(self.n_channels) +
                             " do not match.")

    @property
    def positions(self):
        """Channel positions in the current channel group."""
        return self._positions

    @property
    def channels(self):
        """Channel ids in the current channel group."""
        return self._channels

    @property
    def n_channels(self):
        """Number of channels in the current channel group."""
        return len(self._channels) if self._channels is not None else 0

    @property
    def adjacency(self):
        """Adjacency graph in the current channel group."""
        return self._adjacency

    def change_channel_group(self, group):
        """Change the current channel group."""
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
