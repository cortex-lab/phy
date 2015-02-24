# -*- coding: utf-8 -*-

"""Sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six
from ..utils.array import _as_array


#------------------------------------------------------------------------------
# Sparse CSR
#------------------------------------------------------------------------------

def _csr_from_dense(dense):
    """Create a CSR structure from a dense NumPy array."""
    raise NotImplementedError(("Creating CSR from dense matrix is not "
                               "implemented yet."))


def _check_sparse_components(shape=None, data=None,
                             channels=None, spikes_ptr=None):
    """Ensure the components of a sparse matrix are consistent."""
    if not isinstance(shape, (list, tuple, np.ndarray)):
            raise ValueError("The shape is required and should be an "
                             "array-like list ({0} was given).".format(shape))
    if len(shape) != data.ndim + 1:
        raise ValueError("'shape' {shape} and {ndim}D-array 'data' are "
                         "not consistent.".format(shape=shape,
                                                  ndim=data.ndim))
    if channels.ndim != 1:
        raise ValueError("'channels' should be a 1D array.")
    if spikes_ptr.ndim != 1:
        raise ValueError("'spikes_ptr' should be a 1D array.")
    nitems = data.shape[-1]
    if nitems > np.prod(shape):
        raise ValueError("'data' is too large (n={0:d}) ".format(nitems) +
                         " for the specified shape "
                         "{shape}.".format(shape=shape))
    if len(channels) != (shape[1]):
        raise ValueError(("'channels' should have "
                          "{nexp} elements, "
                          "not {nact}.").format(nexp=(shape[1]),
                                                nact=len(channels)))
    if len(spikes_ptr) != (shape[0] + 1):
        raise ValueError(("'spikes_ptr' should have "
                          "{nexp} elements, "
                          "not {nact}.").format(nexp=(shape[0] + 1),
                                                nact=len(spikes_ptr)))
    if len(data) != len(channels):
        raise ValueError("'data' (n={0:d}) and ".format(len(data)) +
                         "'channels' (n={0:d}) ".format(len(channels)) +
                         "should have the same length")
    return True


class SparseCSR(object):
    """Sparse CSR matrix data structure."""
    def __init__(self, shape=None, data=None, channels=None, spikes_ptr=None):
        # Ensure the arguments are all arrays.
        data = _as_array(data)
        channels = _as_array(channels)
        spikes_ptr = _as_array(spikes_ptr)
        # Ensure the arguments are consistent.
        assert _check_sparse_components(shape=shape,
                                        data=data,
                                        channels=channels,
                                        spikes_ptr=spikes_ptr)
        nitems = data.shape[-1]
        # Structure info.
        self._nitems = nitems
        # Create the structure.
        self._shape = shape
        self._data = data
        self._channels = channels
        self._spikes_ptr = spikes_ptr

    @property
    def shape(self):
        """Shape of the array."""
        return self._shape

    def __eq__(self, other):
        return np.all((self._data == other._data) &
                      (self._channels == other._channels) &
                      (self._spikes_ptr == other._spikes_ptr))

    # I/O methods
    # -------------------------------------------------------------------------

    def save_h5(self, f, path):
        """Save the array in an HDF5 file."""
        f.write_attr(path, 'sparse_type', 'csr')
        f.write_attr(path, 'shape', self._shape)
        f.write(path + '/data', self._data)
        f.write(path + '/channels', self._channels)
        f.write(path + '/spikes_ptr', self._spikes_ptr)

    @staticmethod
    def load_h5(f, path):
        """Load a SparseCSR array from an HDF5 file."""
        f.read_attr(path, 'sparse_type') == 'csr'
        shape = f.read_attr(path, 'shape')
        data = f.read(path + '/data')[...]
        channels = f.read(path + '/channels')[...]
        spikes_ptr = f.read(path + '/spikes_ptr')[...]
        return SparseCSR(shape=shape,
                         data=data,
                         channels=channels,
                         spikes_ptr=spikes_ptr)


def csr_matrix(dense=None, shape=None,
               data=None, channels=None, spikes_ptr=None):
    """Create a CSR matrix from a dense matrix, or from sparse data."""
    if dense is not None:
        # Ensure 'dense' is a ndarray.
        dense = _as_array(dense)
        return _csr_from_dense(dense)
    if data is None or channels is None or spikes_ptr is None:
        raise ValueError("data, channels, and spikes_ptr must be specified.")
    return SparseCSR(shape=shape,
                     data=data, channels=channels, spikes_ptr=spikes_ptr)


def save_h5(f, path, arr, overwrite=False):
    """Save a sparse array into an HDF5 file."""
    if isinstance(arr, SparseCSR):
        arr.save_h5(f, path)
    elif isinstance(arr, np.ndarray):
        f.write(path, arr, overwrite=overwrite)
    else:
        raise ValueError("The array should be a SparseCSR or "
                         "dense NumPy array.")


def load_h5(f, path):
    """Load a sparse array from an HDF5 file."""
    # Sparse array.
    if f.has_attr(path, 'sparse_type'):
        if f.read_attr(path, 'sparse_type') == 'csr':
            return SparseCSR.load_h5(f, path)
        else:
            raise NotImplementedError("Only SparseCSR arrays are implemented "
                                      "currently.")
    # Regular dense dataset.
    else:
        return f.read(path)[...]
