# -*- coding: utf-8 -*-

"""The KwikModel class manages in-memory structures and KWIK file open/save."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from ..ext import six
from .base_model import BaseModel
from ..cluster.manual.cluster_metadata import ClusterMetadata
from .h5 import open_h5
from ..waveform.loader import WaveformLoader
from ..waveform.filter import bandpass_filter, apply_filter
from ..electrode.mea import MEA
from ..utils.logging import debug
from ..utils.array import (PartialArray,
                           _concatenate_virtual_arrays,
                           _as_array,
                           )


#------------------------------------------------------------------------------
# Kwik utility functions
#------------------------------------------------------------------------------

def _to_int_list(l):
    """Convert int strings to ints."""
    return [int(_) for _ in l]


def _list_int_children(group):
    """Return the list of int children of a HDF5 group."""
    return sorted(_to_int_list(group.keys()))


# TODO: refactor the functions below with h5.File.children().

def _list_channel_groups(kwik):
    """Return the list of channel groups in a kwik file."""
    if 'channel_groups' in kwik:
        return _list_int_children(kwik['/channel_groups'])
    else:
        return []


def _list_recordings(kwik):
    """Return the list of recordings in a kwik file."""
    if '/recordings' in kwik:
        recordings = _list_int_children(kwik['/recordings'])
    else:
        recordings = []
    # TODO: return a dictionary of recordings instead of a list of recording
    # ids.
    # return {rec: Bunch({
    #     'start': kwik['/recordings/{0}'.format(rec)].attrs['start_sample']
    # }) for rec in recordings}
    return recordings


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


def _concatenate_spikes(spikes, recs, offsets):
    """Concatenate spike samples belonging to consecutive recordings."""
    assert offsets is not None
    spikes = _as_array(spikes)
    offsets = _as_array(offsets)
    recs = _as_array(recs)
    return (spikes + offsets[recs]).astype(np.uint64)


_COLOR_MAP = np.array([[1., 1., 1.],
                       [1., 0., 0.],
                       [0.5, 0.763, 1.],
                       [0.105, 1., 0.],
                       [1., 0.658, 0.5],
                       [0.421, 0., 1.],
                       [0.5, 1., 0.763],
                       [1., 0.947, 0.],
                       [1., 0.5, 0.974],
                       [0., 0.526, 1.],
                       [0.868, 1., 0.5],
                       [1., 0.316, 0.],
                       [0.553, 0.5, 1.],
                       [0., 1., 0.526],
                       [1., 0.816, 0.5],
                       [1., 0., 0.947],
                       [0.5, 1., 0.921],
                       [0.737, 1., 0.],
                       [1., 0.5, 0.5],
                       [0.105, 0., 1.],
                       [0.553, 1., 0.5],
                       [1., 0.632, 0.],
                       [0.711, 0.5, 1.],
                       [0., 1., 0.842],
                       [1., 0.974, 0.5],
                       [0.9, 0., 0.],
                       [0.45, 0.687, 0.9],
                       [0.095, 0.9, 0.],
                       [0.9, 0.592, 0.45],
                       [0.379, 0., 0.9],
                       [0.45, 0.9, 0.687],
                       [0.9, 0.853, 0.],
                       [0.9, 0.45, 0.876],
                       [0., 0.474, 0.9],
                       [0.782, 0.9, 0.45],
                       [0.9, 0.284, 0.],
                       [0.497, 0.45, 0.9],
                       [0., 0.9, 0.474],
                       [0.9, 0.734, 0.45],
                       [0.9, 0., 0.853],
                       [0.45, 0.9, 0.829],
                       [0.663, 0.9, 0.],
                       [0.9, 0.45, 0.45],
                       [0.095, 0., 0.9],
                       [0.497, 0.9, 0.45],
                       [0.9, 0.568, 0.],
                       [0.639, 0.45, 0.9],
                       [0., 0.9, 0.758],
                       [0.9, 0.876, 0.45]])


_KWIK_EXTENSIONS = ('kwik', 'kwx', 'raw.kwd')


def _kwik_filenames(filename):
    """Return the filenames of the different Kwik files for a given
    experiment."""
    basename, ext = op.splitext(filename)
    return {ext: '{basename}.{ext}'.format(basename=basename, ext=ext)
            for ext in _KWIK_EXTENSIONS}


class SpikeLoader(object):
    """Translate selection with spike ids into selection with
    absolute times."""
    def __init__(self, waveforms, spike_samples):
        self._spike_samples = spike_samples
        # waveforms is a WaveformLoader instance
        self._waveforms = waveforms
        self.dtype = waveforms.dtype
        self.shape = (len(spike_samples),
                      waveforms.n_samples_waveforms,
                      waveforms.n_channels_waveforms)

    def __getitem__(self, item):
        times = self._spike_samples[item]
        return self._waveforms[times]


#------------------------------------------------------------------------------
# KwikModel class
#------------------------------------------------------------------------------

class KwikModel(BaseModel):
    """Holds data contained in a kwik file."""
    def __init__(self, filename=None,
                 channel_group=None,
                 clustering=None):
        super(KwikModel, self).__init__()

        # Initialize fields.
        self._spike_samples = None
        self._spike_clusters = None
        self._metadata = None
        self._clustering = 'main'
        self._probe = None
        self._channels = []
        self._features = None
        self._masks = None
        self._waveforms = None
        self._cluster_metadata = None
        self._traces = None
        self._recording_offsets = None
        self._waveform_loader = None

        # Open the experiment.
        self.filename = filename
        self.open(filename,
                  channel_group=channel_group,
                  clustering=clustering)

    # Internal properties and methods
    # -------------------------------------------------------------------------

    def _check_kwik_version(self):
        # This class only works with kwik version 2 for now.
        kwik_version = self._kwik.read_attr('/', 'kwik_version')
        if kwik_version != 2:
            raise IOError("The kwik version is {v} != 2.".format(kwik_version))

    @property
    def _channel_groups_path(self):
        return '/channel_groups/{0:d}'.format(self._channel_group)

    @property
    def _spikes_path(self):
        return '{0:s}/spikes'.format(self._channel_groups_path)

    @property
    def _channels_path(self):
        return '{0:s}/channels'.format(self._channel_groups_path)

    @property
    def _clusters_path(self):
        return '{0:s}/clusters'.format(self._channel_groups_path)

    def _cluster_path(self, cluster):
        return '{0:s}/{1:d}'.format(self._clustering_path, cluster)

    @property
    def _spike_clusters_path(self):
        return '{0:s}/clusters/{1:s}'.format(self._spikes_path,
                                             self._clustering)

    @property
    def _clustering_path(self):
        return '{0:s}/{1:s}'.format(self._clusters_path, self._clustering)

    # Loading and saving
    # -------------------------------------------------------------------------

    def _open_h5_if_exists(self, file_type, mode=None):
        path = self._filenames[file_type]
        return open_h5(path, mode=mode) if op.exists(path) else None

    def _open_kwik_if_needed(self, mode=None):
        if not self._kwik.is_open():
            self._kwik.open(mode=mode)
            return True
        else:
            if mode is not None:
                self._kwik.mode = mode
            return False

    def _create_waveform_loader(self):
        """Create a waveform loader."""
        n_samples = (self._metadata['extract_s_before'],
                     self._metadata['extract_s_after'])
        order = self._metadata['filter_butter_order']
        b_filter = bandpass_filter(rate=self._metadata['sample_rate'],
                                   low=self._metadata['filter_low'],
                                   high=self._metadata['filter_high'],
                                   order=order)

        def filter(x):
            return apply_filter(x, b_filter)

        self._waveform_loader = WaveformLoader(n_samples=n_samples,
                                               filter=filter,
                                               filter_margin=order * 3,
                                               )

    def _create_cluster_metadata(self):
        self._cluster_metadata = ClusterMetadata()

        @self._cluster_metadata.default
        def group(cluster):
            # Default group is unsorted.
            return 3

    def _load_meta(self):
        """Load metadata from kwik file."""
        metadata = {}
        # Automatically load all metadata from spikedetekt group.
        path = '/application_data/spikedetekt/'
        metadata_fields = self._kwik.attrs(path)
        for field in metadata_fields:
            if field.islower():
                try:
                    metadata[field] = self._kwik.read_attr(path, field)
                except TypeError:
                    debug("Metadata field '{0:s}' not found.".format(field))
        self._metadata = metadata

    def _load_probe(self):
        positions = self._load_channel_positions()
        # TODO: support multiple channel groups.
        self._probe = MEA(positions=positions,
                          n_channels=self.n_channels)

    def _load_recordings(self):
        # Load recordings.
        self._recordings = _list_recordings(self._kwik.h5py_file)
        # This will be updated later if a KWD file is present.
        self._recording_offsets = [0] * (len(self._recordings) + 1)

    def _load_channels(self):
        self._channels = _list_channels(self._kwik.h5py_file,
                                        self._channel_group)

    def _load_channel_groups(self, channel_group=None):
        self._channel_groups = _list_channel_groups(self._kwik.h5py_file)
        if channel_group is None and self._channel_groups:
            # Choose the default channel group if not specified.
            channel_group = self._channel_groups[0]
        # Load the channel group.
        self._channel_group = channel_group
        self._channel_group_changed(channel_group)

    def _load_channel_positions(self):
        """Load the channel positions from the kwik file."""
        positions = []
        for channel in self.channels:
            path = '{0:s}/{1:d}'.format(self._channels_path, channel)
            position = self._kwik.read_attr(path, 'position')
            positions.append(position)
        return np.array(positions)

    def _load_features_masks(self):

        # Load features masks.
        path = '{0:s}/features_masks'.format(self._channel_groups_path)

        if self._kwx is not None:
            fm = self._kwx.read(path)
            self._features_masks = fm
            self._features = PartialArray(fm, 0)

            nfpc = self._metadata['nfeatures_per_channel']
            nc = self.n_channels
            # This partial array simulates a (n_spikes, n_channels) array.
            self._masks = PartialArray(fm,
                                       (slice(0, nfpc * nc, nfpc), 1))
            assert self._masks.shape == (self.n_spikes, nc)

    def _load_spikes(self):
        # Load spike samples.
        path = '{0:s}/time_samples'.format(self._spikes_path)

        # Concatenate the spike samples from consecutive recordings.
        _spikes = self._kwik.read(path)[:]
        self._spike_recordings = self._kwik.read(
            '{0:s}/recording'.format(self._spikes_path))[:]
        self._spike_samples = _concatenate_spikes(_spikes,
                                                  self._spike_recordings,
                                                  self._recording_offsets)

    def _load_spike_clusters(self):
        self._spike_clusters = self._kwik.read(self._spike_clusters_path)[:]

    def _save_spike_clusters(self, spike_clusters):
        assert spike_clusters.shape == self._spike_clusters.shape
        assert spike_clusters.dtype == self._spike_clusters.dtype
        self._spike_clusters = spike_clusters
        sc = self._kwik.read(self._spike_clusters_path)
        sc[:] = spike_clusters

    def _load_clusterings(self, clustering):
        # Once the channel group is loaded, list the clusterings.
        self._clusterings = _list_clusterings(self._kwik.h5py_file,
                                              self.channel_group)
        # Choose the first clustering (should always be 'main').
        if clustering is None and self.clusterings:
            clustering = self.clusterings[0]
        # Load the specified clustering.
        self._clustering = clustering
        self._clustering_changed(clustering)

    def _load_cluster_groups(self):
        clusters = self._kwik.groups(self._clustering_path)
        clusters = [int(cluster) for cluster in clusters]
        for cluster in clusters:
            path = self._cluster_path(cluster)
            group = self._kwik.read_attr(path, 'cluster_group')
            self._cluster_metadata.set_group([cluster], group)

    def _save_cluster_groups(self, cluster_groups):
        assert isinstance(cluster_groups, dict)
        for cluster, group in cluster_groups.items():
            path = self._cluster_path(cluster)
            self._kwik.write_attr(path, 'cluster_group', group)
            self._cluster_metadata.set_group([cluster], group)

    def _load_traces(self):
        if self._kwd is not None:
            i = 0
            self._recording_offsets = []
            traces = []
            for rec in self._recordings:
                path = '/recordings/{0:d}/data'.format(rec)
                data = self._kwd.read(path)
                traces.append(data)
                # NOTE: there is no time gap between the recordings.
                # If a gap were to be added, it should be also added in
                # _concatenate_virtual_arrays() (which doesn't support it
                # currently).
                self._recording_offsets.append(i)
                i += data.shape[0]
            # # Create a new WaveformLoader if needed.
            # if self._waveform_loader is None:
            #     self._create_waveform_loader()
            # Virtual concatenation of the arrays.
            self._traces = _concatenate_virtual_arrays(traces)
            self._waveform_loader.traces = self._traces
        else:
            self._waveform_loader.traces = np.zeros((0, self.n_channels),
                                                    dtype=np.float32)

    def open(self, filename, channel_group=None, clustering=None):
        """Open a Kwik experiment (.kwik, .kwx, .raw.kwd files)."""

        if filename is None:
            raise ValueError("No filename specified.")

        # Open the file.
        self.filename = filename
        self.name = op.splitext(op.basename(filename))[0]

        # Open the files if they exist.
        self._filenames = _kwik_filenames(filename)

        # Open the KWIK file.
        self._kwik = self._open_h5_if_exists('kwik')
        if not self._kwik.is_open():
            raise ValueError("File {0} failed to open.".format(filename))
        self._check_kwik_version()

        # Open the KWX and KWD files.
        self._kwx = self._open_h5_if_exists('kwx')
        self._kwd = self._open_h5_if_exists('raw.kwd')

        # Load the data.
        self._load_meta()
        self._create_waveform_loader()
        self._load_recordings()
        self._load_channel_groups(channel_group)
        self._load_clusterings(clustering)
        self._load_traces()

        # No need to keep the kwik file open.
        self._kwik.close()

    def save(self, spike_clusters, cluster_groups):
        """Commits all in-memory changes to disk."""

        # REFACTOR: with() to open/close the file if needed
        to_close = self._open_kwik_if_needed(mode='a')

        self._save_spike_clusters(spike_clusters)
        self._save_cluster_groups(cluster_groups)

        if to_close:
            self._kwik.close()

    # Changing channel group and clustering
    # -------------------------------------------------------------------------

    def _channel_group_changed(self, value):
        """Called when the channel group changes."""
        if value not in self.channel_groups:
            raise ValueError("The channel group {0} is invalid.".format(value))
        self._channel_group = value

        # Load data.
        _to_close = self._open_kwik_if_needed()
        self._load_channels()
        self._load_spikes()
        self._load_features_masks()
        self._load_probe()
        if _to_close:
            self._kwik.close()

        # Update the list of channels for the waveform loader.
        self._waveform_loader.channels = self._channels

    def _clustering_changed(self, value):
        """Called when the clustering changes."""
        if value not in self.clusterings:
            raise ValueError("The clustering {0} is invalid.".format(value))
        self._clustering = value

        # Load data.
        _to_close = self._open_kwik_if_needed()
        self._create_cluster_metadata()
        self._load_spike_clusters()
        self._load_cluster_groups()
        if _to_close:
            self._kwik.close()

    # Data
    # -------------------------------------------------------------------------

    @property
    def channel_groups(self):
        return self._channel_groups

    @property
    def n_features_per_channel(self):
        return self._metadata['nfeatures_per_channel']

    @property
    def channels(self):
        """List of channels in the current channel group."""
        # TODO: rename to channel_ids?
        return self._channels

    @property
    def n_channels(self):
        """Number of channels in the current channel group."""
        return len(self._channels)

    @property
    def recordings(self):
        return self._recordings

    @property
    def n_recordings(self):
        return len(self._recordings)

    @property
    def clusterings(self):
        return self._clusterings

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        self._clustering_changed(value)

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment."""
        return self._metadata

    @property
    def probe(self):
        """A Probe instance."""
        return self._probe

    @property
    def traces(self):
        """Traces (memory-mapped)."""
        return self._traces

    @property
    def spike_samples(self):
        """Spike samples from the current channel_group."""
        return self._spike_samples

    @property
    def spike_times(self):
        """Spike times (in seconds) from the current channel_group.
        The spike times of all recordings are concatenated."""
        sr = float(self._metadata['sample_rate'])
        return self._spike_samples.astype(np.float64) / sr

    @property
    def spike_recordings(self):
        return self._spike_recordings

    @property
    def n_spikes(self):
        """Return the number of spikes."""
        return len(self._spike_samples)

    @property
    def features(self):
        """Features from the current channel_group (may be memory-mapped)."""
        return self._features

    @property
    def masks(self):
        """Masks from the current channel_group (may be memory-mapped)."""
        return self._masks

    @property
    def features_masks(self):
        """Features-masks from the current channel_group."""
        return self._features_masks

    @property
    def waveforms(self):
        """Waveforms from the current channel_group (may be memory-mapped)."""
        return SpikeLoader(self._waveform_loader, self.spike_samples)

    @property
    def spike_clusters(self):
        """Spike clusters from the current channel_group."""
        return self._spike_clusters

    @property
    def cluster_metadata(self):
        """ClusterMetadata instance holding information about the clusters."""
        return self._cluster_metadata

    # Close
    # -------------------------------------------------------------------------

    def close(self):
        """Close all opened files."""
        if self._kwx is not None:
            self._kwx.close()
        if self._kwd is not None:
            self._kwd.close()
        self._kwik.close()
