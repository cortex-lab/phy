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
    def __init__(self, data=None, channels=None, spikes_ptr=None):
        self._data = data
        self._channels = channels
        self._spikes_ptr = spikes_ptr


def csr_matrix(dense=None, data=None, channels=None, spikes_ptr=None):
    """Create a CSR matrix from a dense matrix, or from sparse data."""
    if dense is not None:
        # Ensure 'dense' is a ndarray.
        if not isinstance(dense, np.ndarray):
            dense = np.array(dense)
        return _csr_from_dense(dense)
    if data is None or channels is None or spikes_ptr is None:
        raise ValueError("data, channels, and spikes_ptr must be specified.")
    return SparseCSR(data=data, channels=channels, spikes_ptr=spikes_ptr)
