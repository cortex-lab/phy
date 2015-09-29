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
    traces = traces[:20000]
    n_samples, n_channels = traces.shape
    sample_rate = 20000
    probe = load_probe('1x32_buzsaki')
    channel_mapping = {i: i for i in range(1, 21, 2)}
    # channel_mapping = None

    sd = SpikeDetector()
    sd.use_single_threshold = False
    sd.set_metadata(probe,
                    channel_mapping=channel_mapping,
                    sample_rate=sample_rate)
    spike_samples, masks = sd.detect(traces)

    # from vispy.app import run
    # from phy.plot import plot_traces
    # plot_traces(sd.subset_traces(traces),
    #             spike_samples=spike_samples,
    #             masks=masks,
    #             n_samples_per_spike=40)
    # run()
