# -*- coding: utf-8 -*-

"""Selector structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Selector class
#------------------------------------------------------------------------------

class Selector(object):
    """Object representing a selection of spikes or clusters."""
    def __init__(self, spike_clusters, n_spikes_max=None):
        self._spike_clusters = spike_clusters

    @property
    def selected_spikes(self):
        """Labels of the selected spikes."""
        return self._selected_spikes

    @selected_spikes.setter
    def selected_spikes(self, value):
        self._selected_spikes = value

    @property
    def selected_clusters(self):
        """Clusters containing at least one selected spike."""
        return self._selected_clusters

    @selected_clusters.setter
    def selected_clusters(self, value):
        self._selected_clusters = value
