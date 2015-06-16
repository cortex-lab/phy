
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
    thresholder = Thresholder(mode='positive',
                              thresholds=strong)
    ae(thresholder(data), data > strong)
