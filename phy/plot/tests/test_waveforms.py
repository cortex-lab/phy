# -*- coding: utf-8 -*-

"""Test waveform plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy import app

from ...utils.logging import set_level
from ..waveforms import Waveforms, WaveformView


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    pass


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_waveforms():

    # TODO: put this in mock dataset module.
    channel_positions = np.array([(35, 310),
                                  (-34, 300),
                                  (33, 290),
                                  (-32, 280),
                                  (31, 270),
                                  (-30, 260),
                                  (29, 250),
                                  (-28, 240),
                                  (27, 230),
                                  (-26, 220),
                                  (25, 210),
                                  (-24, 200),
                                  (23, 190),
                                  (-22, 180),
                                  (21, 170),
                                  (-20, 160),
                                  (19, 150),
                                  (-18, 140),
                                  (17, 130),
                                  (-16, 120),
                                  (15, 110),
                                  (-14, 100),
                                  (13, 90),
                                  (-12, 80),
                                  (11, 70),
                                  (-10, 60),
                                  (9, 50),
                                  (-8, 40),
                                  (7, 30),
                                  (-6, 20),
                                  (5, 10),
                                  (0, 0)], dtype=np.float32)

    # TODO: refactor this
    channel_positions -= channel_positions.min(axis=0)
    channel_positions /= channel_positions.max(axis=0)
    channel_positions = .2 + .6 * channel_positions

    n_clusters = 2
    n_channels = 32
    n_samples = 40
    n_spikes = 10

    waveforms = .25 * np.random.randn(n_spikes, n_channels,
                                      n_samples).astype(np.float32)

    cluster_colors = np.random.uniform(size=(n_clusters, 3),
                                       low=.5, high=.9).astype(np.float32)
    cluster_metadata = {cluster: {'color': color}
                        for cluster, color in enumerate(cluster_colors)}

    spike_clusters = np.random.randint(size=n_spikes,
                                       low=0,
                                       high=n_clusters).astype(np.int32)

    c = WaveformView()
    c.visual.waveforms = waveforms
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_metadata = cluster_metadata
    c.visual.channel_positions = channel_positions

    c.show()
    app.run()
