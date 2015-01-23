# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _unique(x):
    """Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    if len(x) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.bincount(x))[0]


def _normalize(positions):
    """Normalize an array into [0, 1]."""
    # TODO: add 'keep_ratio' option.
    positions = positions.astype(np.float32)
    min, max = positions.min(axis=0), positions.max(axis=0)
    positions_n = (positions - min) / (max - min)
    return positions_n
