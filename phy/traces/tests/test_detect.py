
# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ..detect import Thresholder
from ...io.mock import artificial_traces


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_thresholder():
    n_samples, n_channels = 100, 12
    strong, weak = .1, .2

    data = artificial_traces(n_samples, n_channels)

    # Positive and strong.
    thresholder = Thresholder(mode='positive',
                              thresholds=strong)
    ae(thresholder(data), data > strong)

    # Negative and weak.
    thresholder = Thresholder(mode='negative',
                              thresholds={'weak': weak})
    ae(thresholder(data), data < -weak)
