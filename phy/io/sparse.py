# -*- coding: utf-8 -*-

"""Sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Sparse CSR
#------------------------------------------------------------------------------


def _csr_from_dense(dense):
    """Create a CSR structure from a dense NumPy array."""
    raise NotImplementedError(("Creating CSR from dense matrix is not "
                               "implemented yet."))


class SparseCSR(object):
    """Sparse CSR matrix data structure."""
    def __init__(self, shape=None, data=None, channels=None, spikes_ptr=None):
        # Ensure the arguments are all arrays.
        data = np.asarray(data)
        channels = np.asarray(channels)
        spikes_ptr = np.asarray(spikes_ptr)
        # Ensure the arguments are consistent.
        if not isinstance(shape, tuple):
            raise ValueError("The shape is required and should be a tuple "
                             "({shape} was provided).".format(shape=shape))
        if len(shape) != data.ndim + 1:
            raise ValueError("'shape' {shape} and {ndim}D-array 'data' are "
                             "not consistent.".format(shape=shape,
                                                      ndim=data.ndim))
        if channels.ndim != 1:
            raise ValueError("'channels' should be a 1D array.")
        if spikes_ptr.ndim != 1:
            raise ValueError("'spikes_ptr' should be a 1D array.")
        nitems = data.shape[-1]
        if np.prod(shape) < nitems:
            raise ValueError("'data' is too large for the specified shape "
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
        # Structure info.
        self._nitems = nitems
        # Create the structure.
        self._shape = shape
        self._data = data
        self._channels = channels
        self._spikes_ptr = spikes_ptr


def csr_matrix(dense=None, shape=None,
               data=None, channels=None, spikes_ptr=None):
    """Create a CSR matrix from a dense matrix, or from sparse data."""
    if dense is not None:
        # Ensure 'dense' is a ndarray.
        dense = np.asarray(dense)
        return _csr_from_dense(dense)
    if data is None or channels is None or spikes_ptr is None:
        raise ValueError("data, channels, and spikes_ptr must be specified.")
    return SparseCSR(shape=shape,
                     data=data, channels=channels, spikes_ptr=spikes_ptr)
