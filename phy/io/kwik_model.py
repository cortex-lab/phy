# -*- coding: utf-8 -*-

"""The KwikModel class manages in-memory structures and KWIK file open/save."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six

from .base_model import BaseModel
from .h5 import open_h5
from ..waveform.loader import WaveformLoader

#------------------------------------------------------------------------------
# KwikModel class
#------------------------------------------------------------------------------


class KwikModel(BaseModel):

    def __init__(self, filename=None, channel_group=None, recording=None):

        if filename is not None:
            self.kwik = open_h5(filename)
        else:
            raise ValueError("No filename specified")

        if self.kwik.is_open is False:
            raise ValueError("File {0} failed to open".format(filename))

        self._channel_group = channel_group
        self._recording = recording

    def _channel_group_changed(self, value):
        """Called when the channel group changes."""

        pass

    def _recording_changed(self, value):
        """Called when the recording number changes."""
        pass

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment."""
        raise NotImplementedError()

    @property
    def traces(self):
        """Traces from the current recording (may be memory-mapped)."""
        raise NotImplementedError()

    @property
    def spike_times(self):
        """Spike times from the current channel_group."""
        raise NotImplementedError()

    @property
    def spike_clusters(self):
        """Spike clusters from the current channel_group."""
        raise NotImplementedError()

    @property
    def cluster_metadata(self):
        """ClusterMetadata instance holding information about the clusters."""
        raise NotImplementedError()

    @property
    def features(self):
        """Features from the current channel_group (may be memory-mapped)."""
        raise NotImplementedError()

    @property
    def masks(self):
        """Masks from the current channel_group (may be memory-mapped)."""
        raise NotImplementedError()

    @property
    def waveforms(self):
        """Waveforms from the current channel_group (may be memory-mapped)."""
        raise NotImplementedError()

    @property
    def probe(self):
        """A Probe instance."""
        raise NotImplementedError()

    def save(self):
        """Commits all in-memory changes to disk."""
        pass
