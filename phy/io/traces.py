# -*- coding: utf-8 -*-

"""Raw data readers."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np


#------------------------------------------------------------------------------
# Raw data readers
#------------------------------------------------------------------------------

def _read_recording(filename, rec_name):
    """Open a file and return a recording dataset.

    WARNING: the file is not closed when the function returns, so that the
    memory-mapped array can still be accessed from disk.

    """
    from h5py import File
    f = File(filename, mode='r')
    return f['/recordings/{}/data'.format(rec_name)]


def read_kwd(filename):
    """Read all traces in a `.kwd` file."""
    from h5py import File
    from dask.array import Array

    with File(filename, mode='r') as f:
        rec_names = sorted([name for name in f['/recordings']])
        shapes = [f['/recordings/{}/data'.format(name)].shape
                  for name in rec_names]

    # Create the dask graph for all recordings from the .kwdd file.
    dask = {('data', idx, 0): (_read_recording, filename, rec_name)
            for (idx, rec_name) in enumerate(rec_names)}

    # Make sure all recordings have the same number of channels.
    n_cols = shapes[0][1]
    assert all(shape[1] == n_cols for shape in shapes)

    # Create the dask Array.
    chunks = (tuple(shape[0] for shape in shapes), (n_cols,))
    return Array(dask, 'data', chunks)


def _dat_n_samples(filename, dtype=None, n_channels=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    n_samples = op.getsize(filename) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def read_dat(filename, dtype=None, shape=None, offset=0, n_channels=None):
    """Read traces from a flat binary `.dat` file.

    The output is a memory-mapped file.

    Parameters
    ----------

    filename : str
        The path to the `.dat` file.
    dtype : dtype
        The NumPy dtype.
    offset : 0
        The header size.
    n_channels : int
        The number of channels in the data.
    shape : tuple (optional)
        The array shape. Typically `(n_samples, n_channels)`. The shape is
        automatically computed from the file size if the number of channels
        and dtype are specified.

    """
    if shape is None:
        assert n_channels > 0
        n_samples = _dat_n_samples(filename, dtype=dtype,
                                   n_channels=n_channels)
        shape = (n_samples, n_channels)
    return np.memmap(filename, dtype=dtype, shape=shape,
                     mode='r', offset=offset)


def read_ns5(filename):
    # TODO
    raise NotImplementedError()
