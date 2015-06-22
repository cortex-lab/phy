# -*- coding: utf-8 -*-

"""Kwik creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from h5py import Dataset

from ..h5 import open_h5
from ...utils._types import _as_array
from ...ext.six import string_types, next
from ...ext.six.moves import zip


#------------------------------------------------------------------------------
# Kwik creator
#------------------------------------------------------------------------------

def _first(gen):
    try:
        return next(gen)
    except StopIteration:
        return


def _write_by_chunk(dset, arrs):
    assert isinstance(dset, Dataset)
    first = _first(arrs)
    if first is None:
        return
    # Check the consistency of the first array with the dataset.
    dtype = first.dtype
    shape = first.shape[1:]
    n = first.shape[0]
    assert dset.dtype == dtype
    assert dset.shape[1:] == shape[1:]

    # Copy the first chunk.
    dset[:n, ...] = first
    # # Note: the first has already been iterated.
    # for arr in arrs:
    #     assert isinstance(arr, np.ndarray)
    #     assert arr.dtype == dtype
    #     assert arr.shape[1:] == shape
    #     n += arr.shape[0]
    # # Check the consistency of the HDF5 array with the list of arrays.
    # assert dset.shape[0] == n

    # Start the data copy *from the second array*.
    offset = n
    for arr in arrs:
        n = arr.shape[0]
        arr = arr[...]
        # Match the shape of the chunk array with the dset shape.
        assert arr.shape == (n,) + dset.shape[1:]
        dset[offset:offset + n, ...] = arr
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
            if (isinstance(features, np.ndarray) and
                    isinstance(masks, np.ndarray)):
                fm[:, :, 0] = transform_f(features)
                fm[:, :, 1] = transform_m(masks)
            elif (isinstance(features, list) and
                    isinstance(masks, list)):
                # Concatenate the features/masks chunks in a generator.
                fm_arrs = (np.dstack((transform_f(f), transform_m(m)))
                           for (f, m) in zip(features, masks))
                _write_by_chunk(fm, fm_arrs)
