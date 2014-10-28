# -*- coding: utf-8 -*-

"""Tests of mock datasets."""

import numpy as np

from ..mock import artificial_waveforms


def test_artificial(plot=False):
    nspikes = 10
    nsamples = 32
    nchannels = 64

    waveforms = artificial_waveforms(nspikes=nspikes,
                                     nsamples=nsamples,
                                     nchannels=nchannels)
    assert waveforms.shape == (nspikes, nsamples, nchannels)

    # import matplotlib.pyplot as plt
    # plt.plot(waveforms[..., 0].T)
    # plt.show()
