# -*- coding: utf-8 -*-

"""The KwikModel class manages in-memory structures and Kwik file open/save."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from random import randint
import os

import numpy as np

from .creator import KwikCreator
from ...ext import six
from ..base import BaseModel, ClusterMetadata
from ..h5 import open_h5, File
from ..traces import read_dat, _dat_n_samples
from ...traces.waveform import WaveformLoader, SpikeLoader
from ...traces.filter import bandpass_filter, apply_filter
from ...electrode.mea import MEA
from ...utils.logging import debug, warn
from ...utils.array import (PartialArray,
                            _concatenate_virtual_arrays,
                            _spikes_per_cluster,
                            _unique,
                            )
from ...utils._misc import _read_python
from ...utils._types import _is_integer, _as_array


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
    if path not in kwik:
        return []
    clusterings = sorted(kwik[path].keys())
    # Ensure 'main' is the first if it exists.
    if 'main' in clusterings:
        clusterings.remove('main')
        clusterings = ['main'] + clusterings
    return clusterings


def _concatenate_spikes(spikes, recs, offsets):
    """Concatenate spike samples belonging to consecutive recordings."""
    assert offsets is not None
    spikes = _as_array(spikes)
    offsets = _as_array(offsets)
    recs = _as_array(recs)
    return (spikes + offsets[recs]).astype(np.uint64)


def _create_cluster_group(f, group_id, name,
                          clustering=None,
                          channel_group=None,
                          write_color=True,
                          ):
    cg_path = ('/channel_groups/{0:d}/'
               'cluster_groups/{1:s}/{2:d}').format(channel_group,
                                                    clustering,
                                                    group_id,
                                                    )
    kv_path = cg_path + '/application_data/klustaviewa'
    f.write_attr(cg_path, 'name', name)
    if write_color:
        f.write_attr(kv_path, 'color', randint(2, 10))


def _create_clustering(f, name,
                       channel_group=None,
                       spike_clusters=None,
                       cluster_groups=None,
                       ):
    if cluster_groups is None:
        cluster_groups = {}
    assert isinstance(f, File)
    path = '/channel_groups/{0:d}/spikes/clusters/{1:s}'.format(channel_group,
                                                                name)
    assert not f.exists(path)

    # Save spike_clusters.
    f.write(path, spike_clusters.astype(np.int32))

    cluster_ids = _unique(spike_clusters)

    # Create cluster metadata.
    for cluster in cluster_ids:
        cluster_path = '/channel_groups/{0:d}/clusters/{1:s}/{2:d}'.format(
            channel_group, name, cluster)
        kv_path = cluster_path + '/application_data/klustaviewa'

        # Default group: unsorted.
        cluster_group = cluster_groups.get(cluster, 3)
        f.write_attr(cluster_path, 'cluster_group', cluster_group)
        f.write_attr(kv_path, 'color', randint(2, 10))

    # Create cluster group metadata.
    for group_id, cg_name in _DEFAULT_GROUPS:
        _create_cluster_group(f, group_id, cg_name,
                              clustering=name,
                              channel_group=channel_group,
                              )


def list_kwik(folders):
    """Return the list of Kwik files found in a list of folders."""
    ret = []
    for d in folders:
        for root, dirs, files in os.walk(os.path.expanduser(d)):
            for f in files:
                if f.endswith(".kwik"):
                    ret.append(os.path.join(root, f))
    return ret


def _open_h5_if_exists(kwik_path, file_type, mode=None):
    basename, ext = op.splitext(kwik_path)
    path = '{basename}.{ext}'.format(basename=basename, ext=file_type)
    return open_h5(path, mode=mode) if op.exists(path) else None


def _read_traces(kwik, kwd=None, dtype=None, n_channels=None):
    if '/recordings' not in kwik:
        return
    recordings = kwik.children('/recordings')
    traces = []
    for recording in recordings:
        path = '/recordings/{}/raw'.format(recording)
        if kwik.has_attr(path, 'hdf5_path'):
            if kwd is None:
                return
            traces.append(kwd.read('/recordings/{}/data'.format(recording)))
        elif kwik.has_attr(path, 'dat_path'):
            assert dtype is not None
            assert n_channels
            dat_path = kwik.read_attr(path, 'dat_path')
            # Fallback to relative path if needed (#488).
            if not op.exists(dat_path):
                rel_path = op.basename(dat_path)
                rel_path = op.join(op.dirname(op.realpath(kwik.filename)),
                                   rel_path)
                debug("`{}` doesn't exist, fallback to `{}`.".format(dat_path,
                                                                     rel_path))
                dat_path = rel_path
            n_samples = _dat_n_samples(dat_path,
                                       n_channels=n_channels,
                                       dtype=dtype,
                                       )
            traces.append(read_dat(dat_path,
                                   dtype=dtype,
                                   shape=(n_samples, n_channels)))
    return traces


_DEFAULT_GROUPS = [(0, 'Noise'),
                   (1, 'MUA'),
                   (2, 'Good'),
                   (3, 'Unsorted'),
                   ]


"""Metadata fields that must be provided when creating the Kwik file."""
_mandatory_metadata_fields = ('dtype',
                              'n_channels',
                              'prb_file',
                              'raw_data_files',
                              )


def cluster_group_id(name_or_id):
    """Return the id of a cluster group from its name."""
    if isinstance(name_or_id, six.string_types):
        d = {group.lower(): id for id, group in _DEFAULT_GROUPS}
        return d[name_or_id.lower()]
    else:
        assert _is_integer(name_or_id)
        return name_or_id


#------------------------------------------------------------------------------
# KwikModel class
#------------------------------------------------------------------------------

class KwikModel(BaseModel):
    """Holds data contained in a kwik file."""

    """Names of the default cluster groups."""
    default_cluster_groups = dict(_DEFAULT_GROUPS)

    def __init__(self, kwik_path=None,
                 channel_group=None,
                 clustering=None,
                 waveform_filter=True,
                 ):
        super(KwikModel, self).__init__()

        # Initialize fields.
        self._spike_samples = None
        self._spike_clusters = None
        self._spikes_per_cluster = None
        self._metadata = None
        self._clustering = clustering or 'main'
        self._probe = None
        self._channels = []
        self._channel_order = None
        self._features = None
        self._features_masks = None
        self._masks = None
        self._waveforms = None
        self._cluster_metadata = None
        self._clustering_metadata = {}
        self._traces = None
        self._recording_offsets = None
        self._waveform_loader = None
        self._waveform_filter = waveform_filter

        # Open the experiment.
        self.kwik_path = kwik_path
        self.open(kwik_path,
                  channel_group=channel_group,
                  clustering=clustering)

    @property
    def path(self):
        return self.kwik_path

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

    def _open_kwik_if_needed(self, mode=None):
        if not self._kwik.is_open():
            self._kwik.open(mode=mode)
            return True
        else:
            if mode is not None:
                self._kwik.mode = mode
            return False

    @property
    def n_samples_waveforms(self):
        return (self._metadata['extract_s_before'] +
                self._metadata['extract_s_after'])

    def _create_waveform_loader(self):
        """Create a waveform loader."""
        n_samples = (self._metadata['extract_s_before'],
                     self._metadata['extract_s_after'])
        order = self._metadata['filter_butter_order']
        rate = self._metadata['sample_rate']
        low = self._metadata['filter_low']
        high = self._metadata['filter_high_factor'] * rate
        b_filter = bandpass_filter(rate=rate,
                                   low=low,
                                   high=high,
                                   order=order)

        if self._metadata.get('waveform_filter', True):
            debug("Enable waveform filter.")

            def filter(x):
                return apply_filter(x, b_filter)

            filter_margin = order * 3
        else:
            debug("Disable waveform filter.")
            filter = None
            filter_margin = 0

        dc_offset = self._metadata.get('waveform_dc_offset', None)
        scale_factor = self._metadata.get('waveform_scale_factor', None)
        self._waveform_loader = WaveformLoader(n_samples=n_samples,
                                               filter=filter,
                                               filter_margin=filter_margin,
                                               dc_offset=dc_offset,
                                               scale_factor=scale_factor,
                                               )

    def _update_waveform_loader(self):
        if self._traces is not None:
            self._waveform_loader.traces = self._traces
        else:
            self._waveform_loader.traces = np.zeros((0, self.n_channels),
                                                    dtype=np.float32)

        # Update the list of channels for the waveform loader.
        self._waveform_loader.channels = self._channel_order

    def _create_cluster_metadata(self):
        self._cluster_metadata = ClusterMetadata()

        @self._cluster_metadata.default
        def group(cluster):
            # Default group is unsorted.
            return 3

    def _load_meta(self):
        """Load metadata from kwik file."""
        # Automatically load all metadata from spikedetekt group.
        path = '/application_data/spikedetekt/'
        sample_rate = self._kwik.read_attr(path, 'sample_rate')
        # Load default SpikeDetekt settings.
        curdir = op.dirname(op.realpath(__file__))
        default_settings_path = op.join(curdir,
                                        '../../cluster/default_settings.py')
        settings = _read_python(default_settings_path)
        params = settings['spikedetekt']
        params.update(settings['traces'])
        # Update the parameters from the Kwik file.
        for key in params.keys():
            if self._kwik.has_attr(path, key):
                params[key] = self._kwik.read_attr(path, key)
        # Mandatory data parameters that are not in the default settings.
        params['sample_rate'] = sample_rate
        for key in _mandatory_metadata_fields:
            if self._kwik.has_attr(path, key):
                params[key] = self._kwik.read_attr(path, key)
        self._metadata = params

    def _load_probe(self):
        # Re-create the probe from the Kwik file.
        channel_groups = {}
        for group in self._channel_groups:
            cg_p = '/channel_groups/{:d}'.format(group)
            c_p = cg_p + '/channels'
            channels = self._kwik.read_attr(cg_p, 'channel_order')
            graph = self._kwik.read_attr(cg_p, 'adjacency_graph')
            positions = {
                channel: self._kwik.read_attr(c_p + '/' + str(channel),
                                              'position')
                for channel in channels
            }
            channel_groups[group] = {
                'channels': channels,
                'graph': graph,
                'geometry': positions,
            }
        probe = {'channel_groups': channel_groups}
        self._probe = MEA(probe=probe)

    def _load_recordings(self):
        # Load recordings.
        self._recordings = _list_recordings(self._kwik.h5py_file)
        # This will be updated later if a KWD file is present.
        self._recording_offsets = [0] * (len(self._recordings) + 1)

    def _load_channels(self):
        self._channels = np.array(_list_channels(self._kwik.h5py_file,
                                                 self._channel_group))
        self._channel_order = self._probe.channels
        assert set(self._channel_order) <= set(self._channels)

    def _load_channel_groups(self, channel_group=None):
        self._channel_groups = _list_channel_groups(self._kwik.h5py_file)
        if channel_group is None and self._channel_groups:
            # Choose the default channel group if not specified.
            channel_group = self._channel_groups[0]
        # Load the channel group.
        self._channel_group = channel_group

    def _load_features_masks(self):

        # Load features masks.
        path = '{0:s}/features_masks'.format(self._channel_groups_path)

        nfpc = self._metadata['n_features_per_channel']
        nc = len(self.channel_order)

        if self._kwx is not None:
            self._kwx = _open_h5_if_exists(self.kwik_path, 'kwx')
            if path not in self._kwx:
                debug("There are no features and masks in the `.kwx` file.")
                # No need to keep the file open if it is empty.
                self._kwx.close()
                return
            fm = self._kwx.read(path)
            self._features_masks = fm
            self._features = PartialArray(fm, 0)

            # This partial array simulates a (n_spikes, n_channels) array.
            self._masks = PartialArray(fm, (slice(0, nfpc * nc, nfpc), 1))
            assert self._masks.shape == (self.n_spikes, nc)

    def _load_spikes(self):
        # Load spike samples.
        path = '{0:s}/time_samples'.format(self._spikes_path)

        # Concatenate the spike samples from consecutive recordings.
        if path not in self._kwik:
            debug("There are no spikes in the dataset.")
            return
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

    def _load_clusterings(self, clustering=None):
        # Once the channel group is loaded, list the clusterings.
        self._clusterings = _list_clusterings(self._kwik.h5py_file,
                                              self.channel_group)
        # Choose the first clustering (should always be 'main').
        if clustering is None and self.clusterings:
            clustering = self.clusterings[0]
        # Load the specified clustering.
        self._clustering = clustering

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

    def _load_clustering_metadata(self):
        attrs = self._kwik.attrs(self._clustering_path)
        metadata = {}
        for attr in attrs:
            try:
                metadata[attr] = self._kwik.read_attr(self._clustering_path,
                                                      attr)
            except OSError:
                debug("Error when reading `{}:{}`.".format(
                      self._clustering_path, attr))
        self._clustering_metadata = metadata

    def _save_clustering_metadata(self, metadata):
        if not metadata:
            return
        assert isinstance(metadata, dict)
        for name, value in metadata.items():
            path = self._clustering_path
            self._kwik.write_attr(path, name, value)
        self._clustering_metadata.update(metadata)

    def _load_traces(self):
        n_channels = self._metadata.get('n_channels', None)
        dtype = self._metadata.get('dtype', None)
        dtype = np.dtype(dtype) if dtype else None
        traces = _read_traces(self._kwik,
                              kwd=self._kwd,
                              dtype=dtype,
                              n_channels=n_channels)
        if traces is None:
            return
        # Set the recordings offsets (no delay between consecutive recordings).
        i = 0
        self._recording_offsets = []
        for trace in traces:
            self._recording_offsets.append(i)
            i += trace.shape[0]
        self._traces = _concatenate_virtual_arrays(traces)

    def has_kwx(self):
        """Returns whether the `.kwx` file is present.

        If not, the features and masks won't be available.

        """
        return self._kwx is not None

    def open(self, kwik_path, channel_group=None, clustering=None):
        """Open a Kwik dataset.

        The `.kwik`, `.kwx`, and `.raw.kwd` must be in the same folder with the
        same basename.

        Notes
        -----

        The `.kwik` file is opened in read-only mode, and is automatically
        closed when this function returns. It is temporarily reopened when
        the channel group or clustering changes.

        The `.kwik` file is temporarily opened in append mode when saving.

        The `.kwx` and `.raw.kwd` files stay open in read-only mode as long
        as `model.close()` is not called. This is because there might be
        read accesses to `features_masks` (`.kwx`) and waveforms (`.raw.kwd`)
        while the dataset is opened.

        Parameters
        ----------

        kwik_path : str
            Path to a `.kwik` file.
        channel_group : int or None (default is None)
            The channel group (shank) index to use. This can be changed
            later after the file has been opened. By default, the first
            channel group is used.
        clustering : str or None (default is None)
            The clustering to use. This can be changed later after the file
            has been opened. By default, the `main` clustering is used. An
            error is raised if the `main` clustering doesn't exist.

        """

        if kwik_path is None:
            raise ValueError("No kwik_path specified.")

        # Open the file.
        kwik_path = op.realpath(kwik_path)
        self.kwik_path = kwik_path
        self.name = op.splitext(op.basename(kwik_path))[0]

        # Open the KWIK file.
        self._kwik = _open_h5_if_exists(kwik_path, 'kwik')
        if self._kwik is None:
            raise IOError("File `{0}` doesn't exist.".format(kwik_path))
        if not self._kwik.is_open():
            raise IOError("File `{0}` failed to open.".format(kwik_path))
        self._check_kwik_version()

        # Open the KWX and KWD files.
        self._kwx = _open_h5_if_exists(kwik_path, 'kwx')
        if self._kwx is None:
            warn("The `.kwx` file hasn't been found. "
                 "Features and masks won't be available.")
        self._kwd = _open_h5_if_exists(kwik_path, 'raw.kwd')
        if self._kwd is None:
            debug("The `.raw.kwd` file hasn't been found. "
                  "Traces and waveforms won't be available.")

        # KwikCreator instance.
        self.creator = KwikCreator(kwik_path=kwik_path)

        # Load the data.
        self._load_meta()

        # This needs metadata.
        self._create_waveform_loader()

        self._load_recordings()

        # This generates the recording offset.
        self._load_traces()

        self._load_channel_groups(channel_group)

        # Load the probe.
        self._load_probe()

        # This needs channel groups.
        self._load_clusterings(clustering)

        # This needs the recording offsets.
        # This loads channels, channel_order, spikes, probe.
        self._channel_group_changed(self._channel_group)

        # This loads spike clusters and cluster groups.
        self._clustering_changed(self._clustering)

        # This needs channels, channel_order, and waveform loader.
        self._update_waveform_loader()

        # No need to keep the kwik file open.
        self._kwik.close()

    def save(self, spike_clusters, cluster_groups, clustering_metadata=None):
        """Save the spike clusters and cluster groups in the Kwik file."""

        # REFACTOR: with() to open/close the file if needed
        to_close = self._open_kwik_if_needed(mode='a')

        self._save_spike_clusters(spike_clusters)
        self._save_cluster_groups(cluster_groups)
        self._save_clustering_metadata(clustering_metadata)

        if to_close:
            self._kwik.close()

    def describe(self):
        """Display information about the dataset."""
        def _print(name, value):
            print("{0: <24}{1}".format(name, value))
        _print("Kwik file", self.kwik_path)
        _print("Recordings", self.n_recordings)

        # List of channel groups.
        cg = ['{:d}'.format(g) + ('*' if g == self.channel_group else '')
              for g in self.channel_groups]
        _print("List of shanks", ', '.join(cg))

        # List of clusterings.
        cl = ['{:s}'.format(c) + ('*' if c == self.clustering else '')
              for c in self.clusterings]
        _print("Clusterings", ', '.join(cl))

        _print("Channels", self.n_channels)
        _print("Spikes", self.n_spikes)
        _print("Clusters", self.n_clusters)
        _print("Duration", "{:.0f}s".format(self.duration))

    # Changing channel group and clustering
    # -------------------------------------------------------------------------

    def _channel_group_changed(self, value):
        """Called when the channel group changes."""
        if value not in self.channel_groups:
            raise ValueError("The channel group {0} is invalid.".format(value))
        self._channel_group = value

        # Load data.
        _to_close = self._open_kwik_if_needed()
        self._probe.change_channel_group(value)
        self._load_channels()
        self._load_spikes()
        self._load_features_masks()
        if _to_close:
            self._kwik.close()

        # Update the list of channels for the waveform loader.
        self._waveform_loader.channels = self._channel_order

    def _clustering_changed(self, value):
        """Called when the clustering changes."""
        if value is None:
            return
        if value not in self.clusterings:
            raise ValueError("The clustering {0} is invalid.".format(value))
        self._clustering = value

        # Load data.
        _to_close = self._open_kwik_if_needed()
        self._create_cluster_metadata()
        self._load_spike_clusters()
        self._load_cluster_groups()
        self._load_clustering_metadata()
        if _to_close:
            self._kwik.close()

    # Managing cluster groups
    # -------------------------------------------------------------------------

    def _write_cluster_group(self, group_id, name, write_color=True):
        if group_id <= 3:
            raise ValueError("Default groups cannot be changed.")

        _to_close = self._open_kwik_if_needed(mode='a')

        _create_cluster_group(self._kwik, group_id, name,
                              clustering=self._clustering,
                              channel_group=self._channel_group,
                              write_color=write_color,
                              )

        if _to_close:
            self._kwik.close()

    def add_cluster_group(self, group_id, name):
        """Add a new cluster group."""
        self._write_cluster_group(group_id, name, write_color=True)

    def rename_cluster_group(self, group_id, name):
        """Rename an existing cluster group."""
        self._write_cluster_group(group_id, name, write_color=False)

    def delete_cluster_group(self, group_id):
        if group_id <= 3:
            raise ValueError("Default groups cannot be deleted.")

        path = ('/channel_groups/{0:d}/'
                'cluster_groups/{1:s}/{2:d}').format(self._channel_group,
                                                     self._clustering,
                                                     group_id,
                                                     )

        _to_close = self._open_kwik_if_needed(mode='a')

        self._kwik.delete(path)

        if _to_close:
            self._kwik.close()

    # Managing clusterings
    # -------------------------------------------------------------------------

    def add_clustering(self, name, spike_clusters):
        """Save a new clustering to the file."""
        if name in self._clusterings:
            raise ValueError("The clustering '{0}' ".format(name) +
                             "already exists.")
        assert len(spike_clusters) == self.n_spikes

        _to_close = self._open_kwik_if_needed(mode='a')

        _create_clustering(self._kwik,
                           name,
                           channel_group=self._channel_group,
                           spike_clusters=spike_clusters,
                           )

        # Update the list of clusterings.
        self._load_clusterings(self._clustering)

        if _to_close:
            self._kwik.close()

    def _move_clustering(self, old_name, new_name, copy=None):
        if not copy and old_name == self._clustering:
            raise ValueError("You cannot move the current clustering.")
        if new_name in self._clusterings:
            raise ValueError("The clustering '{0}' ".format(new_name) +
                             "already exists.")

        _to_close = self._open_kwik_if_needed(mode='a')

        if copy:
            func = self._kwik.copy
        else:
            func = self._kwik.move

        # /channel_groups/x/spikes/clusters/<name>
        p = self._spikes_path + '/clusters/'
        func(p + old_name, p + new_name)

        # /channel_groups/x/clusters/<name>
        p = self._clusters_path + '/'
        func(p + old_name, p + new_name)

        # /channel_groups/x/cluster_groups/<name>
        p = self._channel_groups_path + '/cluster_groups/'
        func(p + old_name, p + new_name)

        # Update the list of clusterings.
        self._load_clusterings(self._clustering)

        if _to_close:
            self._kwik.close()

    def rename_clustering(self, old_name, new_name):
        """Rename a clustering in the `.kwik` file."""
        self._move_clustering(old_name, new_name, copy=False)

    def copy_clustering(self, name, new_name):
        """Copy a clustering in the `.kwik` file."""
        self._move_clustering(name, new_name, copy=True)

    def delete_clustering(self, name):
        """Delete a clustering."""
        if name == self._clustering:
            raise ValueError("You cannot delete the current clustering.")
        if name not in self._clusterings:
            raise ValueError(("The clustering {0} "
                              "doesn't exist.").format(name))

        _to_close = self._open_kwik_if_needed(mode='a')

        # /channel_groups/x/spikes/clusters/<name>
        parent = self._kwik.read(self._spikes_path + '/clusters/')
        del parent[name]

        # /channel_groups/x/clusters/<name>
        parent = self._kwik.read(self._clusters_path)
        del parent[name]

        # /channel_groups/x/cluster_groups/<name>
        parent = self._kwik.read(self._channel_groups_path +
                                 '/cluster_groups/')
        del parent[name]

        # Update the list of clusterings.
        self._load_clusterings(self._clustering)

        if _to_close:
            self._kwik.close()

    # Data
    # -------------------------------------------------------------------------

    @property
    def duration(self):
        """Duration of the experiment (in seconds)."""
        if self._traces is None:
            return 0.
        return float(self.traces.shape[0]) / self.sample_rate

    @property
    def channel_groups(self):
        """List of channel groups found in the Kwik file."""
        return self._channel_groups

    @property
    def n_features_per_channel(self):
        """Number of features per channel (generally 3)."""
        return self._metadata['n_features_per_channel']

    @property
    def channels(self):
        """List of all channels in the current channel group.

        This list comes from the /channel_groups HDF5 group in the Kwik file.

        """
        # TODO: rename to channel_ids?
        return self._channels

    @property
    def channel_order(self):
        """List of kept channels in the current channel group.

        If you want the channels used in the features, masks, and waveforms,
        this is the property you want to use, and not `model.channels`.

        The channel order is the same than the one from the PRB file.
        This order was used when generating the features and masks
        in SpikeDetekt2. The same order is used in phy when loading the
        waveforms from the `.raw.kwd` file.

        """
        return self._channel_order

    @property
    def n_channels(self):
        """Number of all channels in the current channel group."""
        return len(self._channels)

    @property
    def recordings(self):
        """List of recording indices found in the Kwik file."""
        return self._recordings

    @property
    def n_recordings(self):
        """Number of recordings found in the Kwik file."""
        return len(self._recordings)

    @property
    def clusterings(self):
        """List of clusterings found in the Kwik file.

        The first one is always `main`.

        """
        return self._clusterings

    @property
    def clustering(self):
        """The currently-active clustering.

        Default is `main`.

        """
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        """Change the currently-active clustering."""
        self._clustering_changed(value)

    @property
    def clustering_metadata(self):
        """A dictionary of key-value metadata specific to the current
        clustering."""
        return self._clustering_metadata

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment.

        This information comes from the PRM file. It was used by
        SpikeDetekt2 and KlustaKwik during automatic clustering.

        """
        return self._metadata

    @property
    def probe(self):
        """A `Probe` instance representing the probe used for the recording.

        This object contains information about the adjacency graph and
        the channel positions.

        """
        return self._probe

    @property
    def traces(self):
        """Raw traces as found in the `.raw.kwd` file.

        This object is memory-mapped to the HDF5 file.

        """
        return self._traces

    @property
    def spike_samples(self):
        """Spike samples from the current channel group.

        This is a NumPy array containing `uint64` values (number of samples
        in unit of the sample rate).

        The spike times of all recordings are concatenated. There is no gap
        between consecutive recordings, currently.

        """
        return self._spike_samples

    @property
    def sample_rate(self):
        """Sample rate of the recording.

        This value is found in the metadata coming from the PRM file.

        """
        return float(self._metadata['sample_rate'])

    @property
    def spike_recordings(self):
        """The recording index for each spike.

        This is a NumPy array of integers with `n_spikes` elements.

        """
        return self._spike_recordings

    @property
    def n_spikes(self):
        """Number of spikes in the current channel group."""
        return (len(self._spike_samples)
                if self._spike_samples is not None else 0)

    @property
    def features(self):
        """Features from the current channel group.

        This is memory-mapped to the `.kwx` file.

        Note: in general, it is better to use the cluster store to access
        the features and masks of some clusters.

        """
        return self._features

    @property
    def masks(self):
        """Masks from the current channel group.

        This is memory-mapped to the `.kwx` file.

        Note: in general, it is better to use the cluster store to access
        the features and masks of some clusters.

        """
        return self._masks

    @property
    def features_masks(self):
        """Features-masks from the current channel group.

        This is memory-mapped to the `.kwx` file.

        Note: in general, it is better to use the cluster store to access
        the features and masks of some clusters.

        """
        return self._features_masks

    @property
    def waveforms(self):
        """High-passed filtered waveforms from the current channel group.

        This is a virtual array mapped to the `.raw.kwd` file. Filtering is
        done on the fly.

        The shape is `(n_spikes, n_samples, n_channels)`.

        """
        return SpikeLoader(self._waveform_loader, self.spike_samples)

    @property
    def spike_clusters(self):
        """Spike clusters from the current channel group and clustering.

        Every element is the cluster identifier of a spike.

        The shape is `(n_spikes,)`.

        """
        return self._spike_clusters

    @property
    def spikes_per_cluster(self):
        """Spikes per cluster from the current channel group and clustering."""
        if self._spikes_per_cluster is None:
            if self._spike_clusters is None:
                self._spikes_per_cluster = {0: self.spike_ids}
            else:
                self._spikes_per_cluster = \
                    _spikes_per_cluster(self.spike_ids, self._spike_clusters)
        return self._spikes_per_cluster

    def update_spikes_per_cluster(self, spc):
        self._spikes_per_cluster = spc

    @property
    def cluster_metadata(self):
        """Metadata about the clusters in the current channel group and
        clustering.

        `cluster_metadata.group(cluster_id)` returns the group of a given
        cluster. The default group is 3 (unsorted).

        """
        return self._cluster_metadata

    @property
    def cluster_ids(self):
        """List of cluster ids from the current channel group and clustering.

        This is a sorted list of unique cluster ids as found in the current
        `spike_clusters` array.

        """
        return _unique(self._spike_clusters)

    @property
    def spike_ids(self):
        """List of spike ids."""
        return np.arange(self.n_spikes, dtype=np.int32)

    @property
    def n_clusters(self):
        """Number of clusters in the current channel group and clustering."""
        return len(self.cluster_ids)

    # Close
    # -------------------------------------------------------------------------

    def close(self):
        """Close the `.kwik`, `.kwx`, and `.raw.kwd` files if they are open."""
        if self._kwx is not None:
            self._kwx.close()
        if self._kwd is not None:
            self._kwd.close()
        self._kwik.close()
