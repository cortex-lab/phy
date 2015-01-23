# -*- coding: utf-8 -*-

"""Tests of array utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ..array import _unique
# from ...datasets.mock import artificial_spike_clusters


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_utils():
    """Test clustering utility functions."""
    _unique([])

    # TODO: uncomment once artificial_spike_clusters is available.
    # n_spikes = 1000
    # n_clusters = 10
    # spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    # assert_array_equal(_unique(spike_clusters), np.arange(n_clusters))
