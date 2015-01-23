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
    """Normalize an array into [-1, 1]."""
    # TODO: add 'keep_ratio' option.
    min, max = positions.min(), positions.max()
    positions_n = (positions - min) / float(max - min)
    positions_n = -1. + 2. * positions_n
    return positions_n
