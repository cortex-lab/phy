# -*- coding: utf-8 -*-

"""Tests of mock datasets."""

import numpy as np

from ..mock import artificial_waveforms


def test_artificial():
    waveforms = artificial_waveforms(nspikes=10, nsamples=32, nchannels=64)
