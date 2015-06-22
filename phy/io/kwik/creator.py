# -*- coding: utf-8 -*-

"""Kwik creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from h5py import Dataset

from ..h5 import open_h5
from ..traces import _dat_n_samples
from ...utils._types import _as_array
from ...utils._misc import _read_python
from ...utils.array import _unique
from ...ext.six import string_types
from ...ext.six.moves import zip


#------------------------------------------------------------------------------
# Kwik creator
#------------------------------------------------------------------------------

def _write_by_chunk(dset, arrs):
    assert isinstance(dset, Dataset)
    if not len(arrs):
        return
    first = arrs[0]
    # Check the consistency of the first array with the dataset.
    dtype = first.dtype
    n = first.shape[0]
    assert dset.dtype == dtype
    assert dset.shape[1:] == first.shape[1:]

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


_DEFAULT_GROUPS = [(0, 'Noise'),
                   (1, 'MUA'),
                   (2, 'Good'),
                   (3, 'Unsorted'),
                   ]


class KwikCreator(object):
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
        with open_h5(self.kwik_path, 'w') as f:
            f.write_attr('/', 'kwik_version', 2)
            f.write_attr('/', 'name', self.basename)

        with open_h5(self.kwx_path, 'w') as f:
            f.write_attr('/', 'kwik_version', 2)

    def set_metadata(self, path, **kwargs):
        assert isinstance(path, string_types)
        assert path
        with open_h5(self.kwik_path, 'a') as f:
            for key, value in kwargs.items():
                f.write_attr(path, key, value)

    def set_probe(self, probe):
        with open_h5(self.kwik_path, 'a') as f:
            probe = probe['channel_groups']
            for group, d in probe.items():
                channels = d['channels']

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
                    # Get the channel position.
                    if channel in positions:
                        position = positions[channel]
                    else:
                        # Default position.
                        position = (0, channel)
                    path = '/channel_groups/{:d}/channels/{:d}'.format(
                        group, channel)

                    f.write_attr(path, 'name', str(channel))
                    f.write_attr(path, 'position', position)

    def add_spikes(self,
                   group=None,
                   spike_samples=None,
                   spike_recordings=None,
                   masks=None,
                   features=None,
                   ):
        assert group >= 0

        spike_samples = _as_array(spike_samples, dtype=np.uint64)
        n_spikes = len(spike_samples)
        if spike_recordings is None:
            spike_recordings = np.zeros(n_spikes, dtype=np.int32)

        # Add spikes in the .kwik file.
        with open_h5(self.kwik_path, 'a') as f:
            # This method can only be called once.
            if '/channel_groups/{:d}/spikes/time_samples'.format(group) in f:
                raise RuntimeError("Spikes have already been added to this "
                                   "dataset.")
            f.write('/channel_groups/{:d}/spikes/time_samples'.format(group),
                    spike_samples)
            f.write('/channel_groups/{:d}/spikes/recording'.format(group),
                    spike_recordings)

        # Add features and masks in the .kwx file.
        assert masks is not None
        assert features is not None

        # Find n_channels and n_features.
        if isinstance(features, np.ndarray):
            _, n_channels, n_features = features.shape
        else:
            assert features
            _, n_channels, n_features = features[0].shape

        # Determine the shape of the features_masks array.
        shape = (n_spikes, n_channels * n_features, 2)

        def transform_f(f):
            return f.reshape((-1, n_channels * n_features))

        def transform_m(m):
            return np.repeat(m, 3, axis=1)

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
                # Concatenate the features/masks chunks in a generator.
                fm_arrs = [np.dstack((transform_f(fet), transform_m(m)))
                           for (fet, m) in zip(features, masks)]
                _write_by_chunk(fm, fm_arrs)

    def add_recording(self, id=None, raw_path=None,
                      start_sample=None, sample_rate=None):
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

    def add_recordings_from_dat(self, files, sample_rate=None,
                                n_channels=None, n_bits=None):
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
                                           n_bits=n_bits)

    def add_recordings_from_kwd(self, file):
        assert file.endswith('.kwd')
        start_sample = 0
        with open_h5(file, 'r') as f:
            recordings = f.children('/recordings')
            for recording in recordings:
                path = '/recordings/{}'.format(recording)
                sample_rate = f.read_attr(path, 'sample_rate')
                self.add_recording(id=int(recording),
                                   start_sample=start_sample,
                                   sample_rate=sample_rate,
                                   raw_data=file,
                                   )
                start_sample += f.read(path + '/data').shape[0]

    def add_cluster_group(self,
                          group=None,
                          id=None,
                          name=None,
                          clustering=None,
                          ):
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
        if cluster_groups is None:
            cluster_groups = {}
        path = '/channel_groups/{0:d}/spikes/clusters/{1:s}'.format(
            group, name)

        with open_h5(self.kwik_path, 'a') as f:
            assert not f.exists(path)

            # Save spike_clusters.
            f.write(path, spike_clusters.astype(np.int32))
            cluster_ids = _unique(spike_clusters)

            # Create cluster metadata.
            for cluster in cluster_ids:
                cluster_path = '/channel_groups/{0:d}/clusters/{1:s}/{2:d}'. \
                    format(group, name, cluster)
                kv_path = cluster_path + '/application_data/klustaviewa'

                # Default group: unsorted.
                cluster_group = cluster_groups.get(cluster, 3)
                f.write_attr(cluster_path, 'cluster_group', cluster_group)
                f.write_attr(kv_path, 'color', np.random.randint(2, 10))

            # Create cluster group metadata.
            for group_id, cg_name in _DEFAULT_GROUPS:
                self.add_cluster_group(id=group_id,
                                       name=cg_name,
                                       clustering=name,
                                       group=group,
                                       )


def create_kwik(prm_file=None, kwik_path=None, probe=None, **kwargs):
    prm = _read_python(prm_file) if prm_file else {}
    sample_rate = kwargs.get('sample_rate', prm.get('sample_rate'))
    assert sample_rate > 0

    # Default SpikeDetekt parameters.
    curdir = op.dirname(op.realpath(__file__))
    default_settings_path = op.join(curdir,
                                    '../../cluster/default_settings.py')
    settings = _read_python(default_settings_path)
    params = settings['spikedetekt_params'](sample_rate)
    # Update with PRM and user parameters.
    params.update(prm)
    params.update(kwargs)

    kwik_path = kwik_path or params['experiment_name'] + '.kwik'
    probe = probe or _read_python(params['prb_file'])

    # KwikCreator.
    creator = KwikCreator(kwik_path)
    creator.create_empty()
    creator.set_probe(probe)
    creator.set_metadata('/application_data/spikedetekt', **params)

    # Add the recordings.
    files = params.get('raw_data_files', [])
    if isinstance(files, string_types):
        if files.endswith('.raw.kwd'):
            creator.add_recordings_from_kwd(files)
        else:
            files = [files]
    if isinstance(files, list) and len(files):
        creator.add_recordings_from_dat(files,
                                        sample_rate=sample_rate,
                                        n_channels=params['n_channels'],
                                        n_bits=params['n_bits'],
                                        )
    return kwik_path
