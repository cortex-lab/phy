# -*- coding: utf-8 -*-

"""The BaseModel class holds the data from an experiment."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# BaseModel class
#------------------------------------------------------------------------------

class BaseModel(object):
    """This class holds data from an experiment.

    This base class must be derived.

    """
    def __init__(self):
        self.name = 'model'
        self._channel_group = None
        self._recording = None
        self._clustering = None

    # Channel groups
    # -------------------------------------------------------------------------

    @property
    def channel_group(self):
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        # The recording is specified by a integer.
        assert isinstance(value, six.integer_types)
        self._channel_group = value
        self._channel_group_changed(value)

    def _channel_group_changed(self, value):
        """Called when the channel group changes.

        May be implemented by child classes.

        """
        pass

    @property
    def channel_groups(self):
        """List of channel groups.

        May be implemented by child classes.

        """
        return []

    # Recordings
    # -------------------------------------------------------------------------

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, value):
        # The recording is specified by a integer.
        assert isinstance(value, six.integer_types)
        self._recording = value
        self._recording_changed(value)

    def _recording_changed(self, value):
        """Called when the recording number changes.

        May be implemented by child classes.

        """
        pass

    @property
    def recordings(self):
        """List of recordings.

        May be implemented by child classes.

        """
        return []

    # Clusterings
    # -------------------------------------------------------------------------

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        # The clustering is specified by a string.
        assert isinstance(value, six.string_types)
        self._clustering = value
        self._clustering_changed(value)

    def _clustering_changed(self, value):
        """Called when the clustering changes.

        May be implemented by child classes.

        """
        pass

    @property
    def clusterings(self):
        """List of clusterings.

        May be implemented by child classes.

        """
        return []

    # Data
    # -------------------------------------------------------------------------

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

    def close(self):
        """Close the model and the underlying files.

        May be implemented by child classes.

        """
        pass
