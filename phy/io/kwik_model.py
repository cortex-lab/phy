# -*- coding: utf-8 -*-

"""The KwikModel class manages in-memory structures and KWIK file open/save."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from ..ext import six
from .base_model import BaseModel
from .h5 import open_h5, _check_hdf5_path
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


def _list_channels(kwik, channel_group=None):
    """Return the list of channels in a kwik file."""
    assert isinstance(channel_group, six.integer_types)
    path = '/channel_groups/{0:d}/channels'.format(channel_group)
    if path in kwik:
        channels = _list_int_children(kwik[path])
        return channels
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


_KWIK_EXTENSIONS = ('kwik', 'kwx')


def _kwik_filenames(filename):
    """Return the filenames of the different Kwik files for a given
    experiment."""
    basename, ext = op.splitext(filename)
    return {ext: '{basename}.{ext}'.format(basename=basename, ext=ext)
            for ext in _KWIK_EXTENSIONS}


class PartialArray(object):
    """Proxy to a view of an array, fixing the last dimension."""
    def __init__(self, arr, col=None):
        self._arr = arr
        self._col = col
        self.dtype = arr.dtype

    @property
    def shape(self):
        return self._arr.shape[:-1]

    def __getitem__(self, item):
        if self._col is None:
            return self._arr[item]
        else:
            if isinstance(item, tuple):
                item += (self._col,)
                return self._arr[item]
            else:
                return self._arr[item, ..., self._col]


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
        self._metadata = None
        self._probe = None
        self._channels = []
        self._features = None
        self._masks = None
        self._waveforms = None
        self._cluster_metadata = None
        self._traces = None

        if filename is None:
            raise ValueError("No filename specified.")

        # Open the file.
        self._kwik = open_h5(filename)
        if not self._kwik.is_open():
            raise ValueError("File {0} failed to open.".format(filename))

        # This class only works with kwik version 2 for now.
        kwik_version = self._kwik.read_attr('/', 'kwik_version')
        if kwik_version != 2:
            raise IOError("The kwik version is {v} != 2.".format(kwik_version))

        # Open the Kwx file if it exists.
        filenames = _kwik_filenames(filename)
        if op.exists(filenames['kwx']):
            self._kwx = open_h5(filenames['kwx'])
        else:
            self._kwx = None

        # Load global information about the file.
        self._load_meta()

        # List channel groups and recordings.
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
        """Load metadata from kwik file."""
        metadata = {}
        # Automatically load all metadata from spikedetekt group.
        path = '/application_data/spikedetekt/'
        metadata_fields = self._kwik.attrs(path)
        for field in metadata_fields:
            metadata[field] = self._kwik.read_attr(path, field)
        # TODO: load probe
        self._metadata = metadata

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

        # Load channels.
        self._channels = _list_channels(self._kwik.h5py_file,
                                        self._channel_group)

        # Load spike times.
        path = '{0:s}/time_samples'.format(self._spikes_path)
        self._spike_times = self._kwik.read(path)

        # Load features masks.
        path = '{0:s}/features_masks'.format(self._channel_groups_path)
        fm = self._kwx.read(path)
        self._features = PartialArray(fm, 0)

        # WARNING: load *all* channel masks in memory for now
        # TODO: sparse, memory mapped, memcache, etc.
        k = self._metadata['nfeatures_per_channel']
        self._masks = fm[:, 0:k * self.n_channels:k, 1]
        assert self._masks.shape == (self.n_spikes, self.n_channels)

        # TODO: load probe

    @property
    def channels(self):
        """List of channels in the current channel group."""
        return self._channels

    @property
    def n_channels(self):
        """Number of channels in the current channel group."""
        return len(self._channels)

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
        return self._metadata

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
    def n_spikes(self):
        """Return the number of spikes."""
        return len(self._spike_times)

    @property
    def features(self):
        """Features from the current channel_group (may be memory-mapped)."""
        return self._features

    @property
    def masks(self):
        """Masks from the current channel_group (may be memory-mapped)."""
        return self._masks

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

    def close(self):
        """Close all opened files."""
        if self._kwx is not None:
            self._kwx.close()
        self._kwik.close()
