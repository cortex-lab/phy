# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac

from phy.utils.datasets import download_test_data
from phy.electrode import load_probe
from ..spike_detect import (SpikeDetector,
                            )


#------------------------------------------------------------------------------
# Test spike detection
#------------------------------------------------------------------------------

def test_detect():

    path = download_test_data('test-32ch-10s.dat')
    traces = np.fromfile(path, dtype=np.int16).reshape((200000, 32))
    traces = traces[:45000]
    n_samples, n_channels = traces.shape
    sample_rate = 20000
    probe = load_probe('1x32_buzsaki')

    sd = SpikeDetector()
    sd.set_metadata(probe)
    spike_samples, masks = sd.detect(traces, sample_rate=sample_rate)
