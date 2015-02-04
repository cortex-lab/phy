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
# Kwik utility functions
#------------------------------------------------------------------------------

def _to_int_list(l):
    """Convert int strings to ints."""
    return [int(_) for _ in l]


def _list_int_children(group):
    """Return the list of int children of a HDF5 group."""
    return sorted(_to_int_list(group.keys()))


def _list_channel_groups(kwik):
    """Return the list of channel groups in a kwik file."""
    if 'channel_groups' in kwik:
        return _list_int_children(kwik['/channel_groups'])
    else:
        return []


def _list_recordings(kwik):
    """Return the list of recordings in a kwik file."""
    if '/recordings' in kwik:
        return _list_int_children(kwik['/recordings'])
    else:
        return []


def _list_clusterings(kwik, channel_group=None):
    """Return the list of clusterings in a kwik file."""
    if channel_group is None:
        raise RuntimeError("channel_group must be specified when listing "
                           "the clusterings.")
    assert isinstance(channel_group, six.integer_types)
    path = '/channel_groups/{0:d}/clusters'.format(channel_group)
    clusterings = sorted(kwik[path].keys())
    # Ensure 'main' exists and is the first.
    assert 'main' in clusterings
    clusterings.remove('main')
    return ['main'] + clusterings


#------------------------------------------------------------------------------
# KwikModel class
#------------------------------------------------------------------------------

class KwikModel(BaseModel):
    """Holds data contained in a kwik file."""
    def __init__(self, filename=None,
                 channel_group=None,
                 recording=None,
                 clustering=None):
        super(KwikModel, self).__init__()

        # Initialize fields.
        self._spike_times = None
        self._spike_clusters = None
        self._probe = None
        self._features = None
        self._masks = None
        self._waveforms = None
        self._cluster_metadata = None
        self._traces = None

        if filename is not None:
            self._kwik = open_h5(filename)
        else:
            raise ValueError("No filename specified.")

        if self._kwik.is_open is False:
            raise ValueError("File {0} failed to open.".format(filename))

        self._load_meta()

        self._channel_groups = _list_channel_groups(self._kwik.h5py_file)
        self._recordings = _list_recordings(self._kwik.h5py_file)

        # Choose the default channel group if not specified.
        if channel_group is None and self.channel_groups:
            channel_group = self.channel_groups[0]
        # Load the channel group.
        self.channel_group = channel_group

        # Choose the default recording if not specified.
        if recording is None and self.recordings:
            recording = self.recordings[0]
        # Load the recording.
        self.recording = recording

        # Once the channel group is loaded, list the clusterings.
        self._clusterings = _list_clusterings(self._kwik.h5py_file,
                                              self.channel_group)
        # Choose the first clustering (should always be 'main').
        if clustering is None and self.clusterings:
            clustering = self.clusterings[0]
        # Load the specified clustering.
        self.clustering = clustering

    # Internal properties and methods
    # -------------------------------------------------------------------------

    @property
    def _channel_groups_path(self):
        return '/channel_groups/{0:d}'.format(self._channel_group)

    @property
    def _spikes_path(self):
        return '{0:s}/spikes'.format(self._channel_groups_path)

    @property
    def _clusters_path(self):
        return '{0:s}/clusters'.format(self._channel_groups_path)

    @property
    def _clustering_path(self):
        return '{0:s}/{1:s}'.format(self._clusters_path, self._clustering)

    def _load_meta(self):
        # TODO: load metadata, probe
        pass

    # Channel group
    # -------------------------------------------------------------------------

    @property
    def channel_groups(self):
        return self._channel_groups

    def _channel_group_changed(self, value):
        """Called when the channel group changes."""
        if value not in self.channel_groups:
            raise ValueError("The channel group {0} is invalid.".format(value))
        self._channel_group = value
        # Load dataset references.
        path = '{0:s}/time_samples'.format(self._spikes_path)
        self._spike_times = self._kwik.read(path)
        # TODO: probe, features, and masks

    @property
    def recordings(self):
        return self._recordings

    def _recording_changed(self, value):
        """Called when the recording number changes."""
        if value not in self.recordings:
            raise ValueError("The recording {0} is invalid.".format(value))
        self._recording = value
        # TODO: traces

    @property
    def clusterings(self):
        return self._clusterings

    def _clustering_changed(self, value):
        """Called when the clustering changes."""
        if value not in self.clusterings:
            raise ValueError("The clustering {0} is invalid.".format(value))
        self._clustering = value
        # NOTE: we are ensured here that self._channel_group is valid.
        path = '{0:s}/clusters/{1:s}'.format(self._spikes_path,
                                             self._clustering)
        self._spike_clusters = self._kwik.read(path)
        # TODO: cluster metadata

    # Data
    # -------------------------------------------------------------------------

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment."""
        raise NotImplementedError()

    @property
    def probe(self):
        """A Probe instance."""
        raise NotImplementedError()

    @property
    def traces(self):
        """Traces from the current recording (may be memory-mapped)."""
        raise NotImplementedError()

    @property
    def spike_times(self):
        """Spike times from the current channel_group."""
        return self._spike_times

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
    def spike_clusters(self):
        """Spike clusters from the current channel_group."""
        return self._spike_clusters

    @property
    def cluster_metadata(self):
        """ClusterMetadata instance holding information about the clusters."""
        raise NotImplementedError()

    def save(self):
        """Commits all in-memory changes to disk."""
        pass
