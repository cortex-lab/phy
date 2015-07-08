# -*- coding: utf-8 -*-

"""Raw data readers."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from ..utils.array import _concatenate_virtual_arrays


#------------------------------------------------------------------------------
# Raw data readers
#------------------------------------------------------------------------------

def read_kwd(kwd_handle):
    """Read all traces in a  `.kwd` file.

    The output is a memory-mapped file.

    """
    f = kwd_handle
    if '/recordings' not in f:
        return
    recordings = f.children('/recordings')
    traces = []
    for recording in recordings:
        traces.append(f.read('/recordings/{}/data'.format(recording)))
    return _concatenate_virtual_arrays(traces)


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


def _dat_n_samples(filename, dtype=None, n_channels=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    n_samples = op.getsize(filename) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def read_ns5(filename):
    # TODO
    raise NotImplementedError()
