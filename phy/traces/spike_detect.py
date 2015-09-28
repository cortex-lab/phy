# -*- coding: utf-8 -*-

"""Spike detection."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from traitlets.config.configurable import Configurable
from traitlets import Int, Float, Unicode, Bool


#------------------------------------------------------------------------------
# SpikeDetector
#------------------------------------------------------------------------------

class SpikeDetector(Configurable):
    filter_low = Float(500.)
    filter_butter_order = Int(3)
    chunk_size_seconds = Float(1)
    chunk_overlap_seconds = Float(.015)
    n_excerpts = Int(50)
    excerpt_size_seconds = Float(1.)
    use_single_threshold = Bool(True)
    threshold_strong_std_factor = Float(4.5)
    threshold_weak_std_factor = Float(2)
    detect_spikes = Unicode('negative')
    connected_component_join_size = Int(1)
    extract_s_before = Int(10)
    extract_s_after = Int(10)
    weight_power = Float(2)

    def __init__(self, ctx=None):
        super(SpikeDetector, self).__init__()
        if not ctx or not hasattr(ctx, 'cache'):
            return
        self.filter = ctx.cache(self.filter)
        self.extract_components = ctx.cache(self.extract_components)
        self.extract_spikes = ctx.cache(self.extract_spikes)

    def set_metadata(self, probe, channel_mapping=None):
        self.probe = probe
        if channel_mapping is None:
            channel_mapping = {c: c for c in probe.channels}
        self.channel_mapping = channel_mapping
        self.channels = probe.channels
        self.n_channels = probe.n_channels

    def filter(self, raw_data):
        pass

    def extract_components(self, filtered):
        pass

    def extract_spikes(self, components):
        return None, None, None

    def detect(self, raw_data, sample_rate=None):
        assert sample_rate > 0
        assert raw_data.ndim == 2
        assert raw_data.shape[1] == self.n_channels

        filtered = self.filter(raw_data)
        components = self.extract_components(filtered)
        spike_samples, masks, waveforms = self.extract_spikes(components)
        return spike_samples, masks
