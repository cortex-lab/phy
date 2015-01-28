# -*- coding: utf-8 -*-

"""The Experiment class holds the data from an experiment."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Experiment class
#------------------------------------------------------------------------------

class BaseExperiment(object):
    """This class holds data from an experiment.

    This base class must be derived.

    """

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment."""
        raise NotImplementedError()

    def traces(self, recording=0):
        """Traces from a given recording (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def spike_times(self, channel_group=0):
        """Spike times from a given channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    def spike_clusters(self, channel_group=0):
        """Spike clusters from a given channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    def cluster_metadata(self, channel_group=0):
        """ClusterMetadata instance holding information about the clusters.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    def features(self, channel_group=0):
        """Features from a given channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def masks(self, channel_group=0):
        """Masks from a given channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def waveforms(self, channel_group=0):
        """Waveforms from a given channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def probe(self):
        """A Probe instance.

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def save(self):
        """Save the data.

        May be implemented by child classes.

        """
        raise NotImplementedError()
