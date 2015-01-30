# -*- coding: utf-8 -*-

"""The Experiment class holds the data from an experiment."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Experiment class
#------------------------------------------------------------------------------

class BaseExperiment(object):
    """This class holds data from an experiment.

    This base class must be derived.

    """
    def __init__(self):
        self._channel_group = None
        self._recording = None

    @property
    def channel_group(self):
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        assert isinstance(value, six.integer_types)
        self._channel_group = value
        self._channel_group_changed(value)

    def _channel_group_changed(self, value):
        """Called when the channel group changes.

        May be implemented by child classes.

        """
        pass

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, value):
        assert isinstance(value, six.integer_types)
        self._recording = value
        self._recording_changed(value)

    def _recording_changed(self, value):
        """Called when the recording number changes.

        May be implemented by child classes.

        """
        pass

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment.

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def traces(self):
        """Traces from the current recording (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def spike_times(self):
        """Spike times from the current channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def spike_clusters(self):
        """Spike clusters from the current channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def cluster_metadata(self):
        """ClusterMetadata instance holding information about the clusters.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def features(self):
        """Features from the current channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def masks(self):
        """Masks from the current channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def waveforms(self):
        """Waveforms from the current channel_group (may be memory-mapped).

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
