# -*- coding: utf-8 -*-

"""Kwik creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import itertools
import types

import numpy as np
from h5py import Dataset

from ...electrode.mea import load_probe
from ..h5 import open_h5
from ..traces import _dat_n_samples
from ...utils._types import _as_array
from ...utils.logging import warn
from ...utils._misc import _read_python
from ...utils.array import _unique
from ...ext.six import string_types, next
from ...ext.six.moves import zip


#------------------------------------------------------------------------------
# Kwik creator
#------------------------------------------------------------------------------

def _write_by_chunk(dset, arrs):
    # Note: arrs should be a generator for performance reasons.
    assert isinstance(dset, Dataset)
    # Start the data.
    offset = 0
    for arr in arrs:
        n = arr.shape[0]
        arr = arr[...]
        # Match the shape of the chunk array with the dset shape.
        assert arr.shape == (n,) + dset.shape[1:]
        dset[offset:offset + n, ...] = arr
        offset += arr.shape[0]
    # Check that the copy is complete.
    assert offset == dset.shape[0]


def _concat(arrs):
    return np.hstack([_[...] for _ in arrs])


_DEFAULT_GROUPS = [(0, 'Noise'),
                   (1, 'MUA'),
                   (2, 'Good'),
                   (3, 'Unsorted'),
                   ]


class KwikCreator(object):
    """Create and modify a `.kwik` file."""
    def __init__(self, basename=None, kwik_path=None, kwx_path=None):
        # Find the .kwik filename.
        if kwik_path is None:
            assert basename is not None
            if basename.endswith('.kwik'):
                basename, _ = op.splitext(basename)
            kwik_path = basename + '.kwik'
        self.kwik_path = kwik_path
        if basename is None:
            basename, _ = op.splitext(kwik_path)
        self.basename = basename

        # Find the .kwx filename.
        if kwx_path is None:
            basename, _ = op.splitext(kwik_path)
            kwx_path = basename + '.kwx'
        self.kwx_path = kwx_path

    def create_empty(self):
        """Create empty `.kwik` and `.kwx` files."""
        assert not op.exists(self.kwik_path)
        with open_h5(self.kwik_path, 'w') as f:
            f.write_attr('/', 'kwik_version', 2)
            f.write_attr('/', 'name', self.basename)

        assert not op.exists(self.kwx_path)
        with open_h5(self.kwx_path, 'w') as f:
            f.write_attr('/', 'kwik_version', 2)

    def set_metadata(self, path, **kwargs):
        """Set metadata fields in a HDF5 path."""
        assert isinstance(path, string_types)
        assert path
        with open_h5(self.kwik_path, 'a') as f:
            for key, value in kwargs.items():
                f.write_attr(path, key, value)

    def set_probe(self, probe):
        """Save a probe dictionary in the file."""
        with open_h5(self.kwik_path, 'a') as f:
            probe = probe['channel_groups']
            for group, d in probe.items():
                group = int(group)
                channels = np.array(list(d['channels']), dtype=np.int32)

                # Write the channel order.
                f.write_attr('/channel_groups/{:d}'.format(group),
                             'channel_order', channels)

                # Write the probe adjacency graph.
                graph = d.get('graph', [])
                graph = np.array(graph, dtype=np.int32)
                f.write_attr('/channel_groups/{:d}'.format(group),
                             'adjacency_graph', graph)

                # Write the channel positions.
                positions = d.get('geometry', {})
                for channel in channels:
                    channel = int(channel)
                    # Get the channel position.
                    if channel in positions:
                        position = positions[channel]
                    else:
                        # Default position.
                        position = (0, channel)
                    path = '/channel_groups/{:d}/channels/{:d}'.format(
                        group, channel)

                    f.write_attr(path, 'name', str(channel))

                    position = np.array(position, dtype=np.float32)
                    f.write_attr(path, 'position', position)

    def add_spikes(self,
                   group=None,
                   spike_samples=None,
                   spike_recordings=None,
                   masks=None,
                   features=None,
                   ):
        """Add spikes in the file.

        Parameters
        ----------

        group : int
            Channel group.
        spike_samples : ndarray
            The spike times in number of samples.
        spike_recordings : ndarray (optional)
            The recording indices of every spike.
        masks : ndarray or list of ndarrays
            The masks.
        features : ndarray or list of ndarrays
            The features.

        Note
        ----

        The features and masks can be passed as lists or generators of
        (memmapped) data chunks in order to avoid loading the entire arrays
        in RAM.

        """
        assert group >= 0

        if isinstance(spike_samples, list):
            spike_samples = _concat(spike_samples)
        spike_samples = _as_array(spike_samples, dtype=np.float64).ravel()
        n_spikes = len(spike_samples)
        if spike_recordings is None:
            spike_recordings = np.zeros(n_spikes, dtype=np.int32)
        spike_recordings = spike_recordings.ravel()

        # Add spikes in the .kwik file.
        assert op.exists(self.kwik_path)
        with open_h5(self.kwik_path, 'a') as f:
            # This method can only be called once.
            if '/channel_groups/{:d}/spikes/time_samples'.format(group) in f:
                raise RuntimeError("Spikes have already been added to this "
                                   "dataset.")
            time_samples = spike_samples.astype(np.uint64)
            frac = ((spike_samples - time_samples) * 255).astype(np.uint8)
            f.write('/channel_groups/{:d}/spikes/time_samples'.format(group),
                    time_samples)
            f.write('/channel_groups/{}/spikes/time_fractional'.format(group),
                    frac)
            f.write('/channel_groups/{:d}/spikes/recording'.format(group),
                    spike_recordings)

        if masks is None and features is None:
            return
        # Add features and masks in the .kwx file.
        assert masks is not None
        assert features is not None

        # Find n_channels and n_features.
        if isinstance(features, np.ndarray):
            _, n_channels, n_features = features.shape
        else:
            assert features
            # Ensure we have a generator.
            if isinstance(features, list):
                features = (_ for _ in features)
            assert isinstance(features, types.GeneratorType)
            # Extract the first item in the generator.
            features_first = next(features)
            _, n_channels, n_features = features_first.shape
            # NOTE: this is equivalent to resetting the generator.
            features = itertools.chain((features_first,), features)

        # Determine the shape of the features_masks array.
        shape = (n_spikes, n_channels * n_features, 2)

        def transform_f(f):
            return f[...].reshape((-1, n_channels * n_features))

        def transform_m(m):
            return np.repeat(m[...], 3, axis=1)

        assert op.exists(self.kwx_path)
        with open_h5(self.kwx_path, 'a') as f:
            fm = f.write('/channel_groups/{:d}/features_masks'.format(group),
                         shape=shape, dtype=np.float32)

            # Write the features and masks in one block.
            if (isinstance(features, np.ndarray) and
                    isinstance(masks, np.ndarray)):
                fm[:, :, 0] = transform_f(features)
                fm[:, :, 1] = transform_m(masks)
            # Write the features and masks chunk by chunk.
            else:
                # Concatenate the features/masks chunks.
                fm_arrs = (np.dstack((transform_f(fet), transform_m(m)))
                           for (fet, m) in zip(features, masks))
                _write_by_chunk(fm, fm_arrs)

    def add_recording(self, id=None, raw_path=None,
                      start_sample=None, sample_rate=None):
        """Add a recording.

        Parameters
        ----------

        id : int
            The recording id (0, 1, 2, etc.).
        raw_path : str
            Path to the file containing the raw data.
        start_sample : int
            The offset of the recording, in number of samples.
        sample_rate : float
            The sample rate of the recording

        """
        path = '/recordings/{:d}'.format(id)
        start_sample = int(start_sample)
        sample_rate = float(sample_rate)

        with open_h5(self.kwik_path, 'a') as f:
            f.write_attr(path, 'name', 'recording_{:d}'.format(id))
            f.write_attr(path, 'start_sample', start_sample)
            f.write_attr(path, 'sample_rate', sample_rate)
            f.write_attr(path, 'start_time', start_sample / sample_rate)
            if raw_path:
                if op.splitext(raw_path)[1] == '.kwd':
                    f.write_attr(path + '/raw', 'hdf5_path', raw_path)
                elif op.splitext(raw_path)[1] == '.dat':
                    f.write_attr(path + '/raw', 'dat_path', raw_path)

    def _add_recordings_from_dat(self, files, sample_rate=None,
                                n_channels=None, dtype=None):
        start_sample = 0
        for i, filename in enumerate(files):
            # WARNING: different sample rates in recordings is not
            # supported yet.
            self.add_recording(id=i,
                               start_sample=start_sample,
                               sample_rate=sample_rate,
                               raw_path=filename,
                               )
            assert op.splitext(filename)[1] == '.dat'
            # Compute the offset for different recordings.
            start_sample += _dat_n_samples(filename,
                                           n_channels=n_channels,
                                           dtype=dtype)

    def _add_recordings_from_kwd(self, file, sample_rate=None):
        assert file.endswith('.kwd')
        start_sample = 0
        with open_h5(file, 'r') as f:
            recordings = f.children('/recordings')
            for recording in recordings:
                path = '/recordings/{}'.format(recording)
                if f.has_attr(path, 'sample_rate'):
                    sample_rate = f.read_attr(path, 'sample_rate')
                assert sample_rate > 0
                self.add_recording(id=int(recording),
                                   start_sample=start_sample,
                                   sample_rate=sample_rate,
                                   raw_path=file,
                                   )
                start_sample += f.read(path + '/data').shape[0]

    def add_cluster_group(self,
                          group=None,
                          id=None,
                          name=None,
                          clustering=None,
                          ):
        """Add a cluster group.

        Parameters
        ----------

        group : int
            The channel group.
        id : int
            The cluster group id.
        name : str
            The cluster group name.
        clustering : str
            The name of the clustering.

        """
        assert group >= 0
        cg_path = ('/channel_groups/{0:d}/'
                   'cluster_groups/{1:s}/{2:d}').format(group,
                                                        clustering,
                                                        id,
                                                        )
        with open_h5(self.kwik_path, 'a') as f:
            f.write_attr(cg_path, 'name', name)

    def add_clustering(self,
                       group=None,
                       name=None,
                       spike_clusters=None,
                       cluster_groups=None,
                       ):
        """Add a clustering.

        Parameters
        ----------

        group : int
            The channel group.
        name : str
            The clustering name.
        spike_clusters : ndarray
            The spike clusters assignements. This is `(n_spikes,)` array.
        cluster_groups : dict
            The cluster group of every cluster.

        """
        if cluster_groups is None:
            cluster_groups = {}
        path = '/channel_groups/{0:d}/spikes/clusters/{1:s}'.format(
            group, name)

        with open_h5(self.kwik_path, 'a') as f:
            assert not f.exists(path)

            # Save spike_clusters.
            spike_clusters = spike_clusters.astype(np.int32).ravel()
            f.write(path, spike_clusters)
            cluster_ids = _unique(spike_clusters)

            # Create cluster metadata.
            for cluster in cluster_ids:
                cluster_path = '/channel_groups/{0:d}/clusters/{1:s}/{2:d}'. \
                    format(group, name, cluster)

                # Default group: unsorted.
                cluster_group = cluster_groups.get(cluster, 3)
                f.write_attr(cluster_path, 'cluster_group', cluster_group)

            # Create cluster group metadata.
            for group_id, cg_name in _DEFAULT_GROUPS:
                self.add_cluster_group(id=group_id,
                                       name=cg_name,
                                       clustering=name,
                                       group=group,
                                       )


def create_kwik(prm_file=None, kwik_path=None, overwrite=False,
                probe=None, **kwargs):
    """Create a new Kwik dataset from a PRM file."""
    prm = _read_python(prm_file) if prm_file else {}

    if prm:
        assert 'spikedetekt' in prm
        assert 'traces' in prm
        sample_rate = prm['traces']['sample_rate']

    if 'sample_rate' in kwargs:
        sample_rate = kwargs['sample_rate']

    assert sample_rate > 0

    # Default SpikeDetekt parameters.
    curdir = op.dirname(op.realpath(__file__))
    default_settings_path = op.join(curdir,
                                    '../../cluster/default_settings.py')
    settings = _read_python(default_settings_path)
    params = settings['spikedetekt']
    params.update(settings['traces'])
    # Update with PRM and user parameters.
    if prm:
        params['experiment_name'] = prm['experiment_name']
        params['prb_file'] = prm['prb_file']
        params.update(prm['spikedetekt'])
        params.update(prm['traces'])
    params.update(kwargs)

    kwik_path = kwik_path or params['experiment_name'] + '.kwik'
    kwx_path = op.splitext(kwik_path)[0] + '.kwx'
    if op.exists(kwik_path):
        if overwrite:
            os.remove(kwik_path)
            os.remove(kwx_path)
        else:
            raise IOError("The `.kwik` file already exists. Please use "
                          "the `--overwrite` option.")

    # Ensure the probe file exists if it is required.
    if probe is None:
        probe = load_probe(params['prb_file'])
    assert probe

    # KwikCreator.
    creator = KwikCreator(kwik_path)
    creator.create_empty()
    creator.set_probe(probe)

    # Add the recordings.
    raw_data_files = params.get('raw_data_files', None)
    if isinstance(raw_data_files, string_types):
        if raw_data_files.endswith('.raw.kwd'):
            creator._add_recordings_from_kwd(raw_data_files,
                                             sample_rate=sample_rate,
                                             )
        else:
            raw_data_files = [raw_data_files]
    if isinstance(raw_data_files, list) and len(raw_data_files):
        # The dtype must be a string so that it can be serialized in HDF5.
        if not params.get('dtype', None):
            warn("The `dtype` parameter is mandatory. Using a default value "
                 "of `int16` for now. Please update your `.prm` file.")
            params['dtype'] = 'int16'
        assert 'dtype' in params and isinstance(params['dtype'], string_types)
        dtype = np.dtype(params['dtype'])
        assert dtype is not None

        # The number of channels in the .dat file *must* be specified.
        n_channels = params['n_channels']
        assert n_channels > 0
        creator._add_recordings_from_dat(raw_data_files,
                                         sample_rate=sample_rate,
                                         n_channels=n_channels,
                                         dtype=dtype,
                                         )

    creator.set_metadata('/application_data/spikedetekt', **params)

    return kwik_path
