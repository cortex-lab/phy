# -*- coding: utf-8 -*-

"""Kwik creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from ..h5 import open_h5
from ...utils._types import _as_array
from ...ext.six import string_types


#------------------------------------------------------------------------------
# Kwik creator
#------------------------------------------------------------------------------

def _write_by_chunk(dset, arrs, transform=None):
    # One can specify a transform function to apply to every chunk before
    # it is copied to the HDF5 dataset.
    if transform is None:
        transform = lambda _: _
    assert dset
    assert isinstance(arrs, list)
    if len(arrs) == 0:
        return
    # Check the consistency of all arrays.
    dtype = arrs[0].dtype
    shape = arrs[0].shape[1:]
    n = arrs[0].shape[0]
    for arr in arrs[1:]:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == dtype
        assert arr.shape[1:] == shape
        n += arr.shape[0]
    # Check the consistency of the HDF5 array with the list of arrays.
    assert dset.shape == (n,) + shape

    # Start the data copy.
    offset = 0
    for arr in arrs:
        n = arr.shape[0]
        dset[offset:offset + n, ...] = transform(arr[...])
        offset += arr.shape[0]
    assert offset == dset.shape[0]


class KwikCreator(object):
    def __init__(self, basename=None, kwik_path=None, kwx_path=None):
        # Find the .kwik filename.
        if kwik_path is None:
            assert basename is not None
            if basename.endswith('.kwik'):
                basename, _ = op.splitext(basename)
            kwik_path = basename + '.kwik'
        if op.exists(kwik_path):
            raise ValueError("The file `{}` already exists.".format(kwik_path))
        self.kwik_path = kwik_path
        if basename is None:
            basename, _ = op.splitext(kwik_path)
        self.basename = basename

        # Find the .kwx filename.
        if kwx_path is None:
            basename, _ = op.splitext(kwik_path)
            kwx_path = basename + '.kwx'
        if op.exists(kwx_path):
            raise ValueError("The file `{}` already exists.".format(kwx_path))
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
        elif isinstance(features, list):
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

            # Write the features either in one block, or chunk by chunk.
            if isinstance(features, np.ndarray):
                fm[:, :, 0] = transform_f(features)
            elif isinstance(features, list):
                _write_by_chunk(fm[..., 0], features, transform_f)

            # Write the masks either in one block, or chunk by chunk.
            if isinstance(masks, np.ndarray):
                fm[:, :, 1] = transform_m(masks)
            elif isinstance(masks, list):
                _write_by_chunk(fm[..., 1], masks, transform_m)
