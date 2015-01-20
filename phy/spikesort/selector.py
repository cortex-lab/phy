# -*- coding: utf-8 -*-

"""Selector structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six
from ..utils.logging import debug, info, warn


#------------------------------------------------------------------------------
# Selector class
#------------------------------------------------------------------------------

def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    return np.nonzero(np.bincount(x))[0]


class Selector(object):
    """Object representing a selection of spikes or clusters."""
    def __init__(self, spike_clusters, n_spikes_max=None):
        self._spike_clusters = spike_clusters
        self._n_spikes_max = n_spikes_max
        self._selected_spikes = np.array([], dtype=np.int64)

    @property
    def n_spikes_max(self):
        """Maximum number of spikes allowed in the selection."""
        return self._n_spikes_max

    @n_spikes_max.setter
    def n_spikes_max(self, value):
        self._n_spikes_max = value

    @property
    def selected_spikes(self):
        """Labels of the selected spikes."""
        return self._selected_spikes

    @selected_spikes.setter
    def selected_spikes(self, value):
        value = np.asarray(value)
        size = len(value)
        size_max = self._n_spikes_max
        if self._n_spikes_max is not None and size > size_max:
            debug("{0:d} spikes were selected whereas ".format(size) +
                  "no more than {0:d} are allowed; ".format(size_max) +
                  "keeping only the first {0:d} now.".format(size_max))
            self._selected_spikes = value[:size_max]
        else:
            self._selected_spikes = value

    @property
    def selected_clusters(self):
        """Clusters containing at least one selected spike."""
        return _unique(self._spike_clusters[self._selected_spikes])

    @selected_clusters.setter
    def selected_clusters(self, value):
        raise NotImplementedError("Cluster selection has not been "
                                  "implemented yet.")
