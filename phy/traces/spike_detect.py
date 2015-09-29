# -*- coding: utf-8 -*-

"""Spike detection."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from traitlets.config.configurable import Configurable
from traitlets import Int, Float, Unicode, Bool

from phy.electrode import MEA
from phy.utils.array import get_excerpts
from .detect import FloodFillDetector, Thresholder, compute_threshold
from .filter import Filter
from .waveform import WaveformExtractor

logger = logging.getLogger(__name__)


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
        self.find_thresholds = ctx.cache(self.find_thresholds)
        self.filter = ctx.cache(self.filter)
        self.extract_components = ctx.cache(self.extract_components)
        self.extract_spikes = ctx.cache(self.extract_spikes)

    def set_metadata(self, probe, channel_mapping=None, sample_rate=None):
        assert isinstance(probe, MEA)
        self.probe = probe

        assert sample_rate > 0
        self.sample_rate = sample_rate

        if channel_mapping is None:
            channel_mapping = {c: c for c in probe.channels}
        self.channel_mapping = channel_mapping
        self.adjacency = self.probe.adjacency

        # Array of channel idx to consider.
        self.channels = sorted(channel_mapping.keys())
        self.n_channels = len(self.channels)
        self.n_samples_waveforms = self.extract_s_before + self.extract_s_after

    def _select_channels(self, traces):
        return traces[:, self.channels]

    def find_thresholds(self, traces):
        """Find weak and strong thresholds in filtered traces."""
        excerpt_size = int(self.excerpt_size_seconds * self.sample_rate)
        single_threshold = self.use_single_threshold
        std_factor = (self.threshold_weak_std_factor,
                      self.threshold_strong_std_factor)

        logger.info("Extracting some data for finding the thresholds...")
        excerpt = get_excerpts(traces, n_excerpts=self.n_excerpts,
                               excerpt_size=excerpt_size)

        logger.info("Filtering the excerpts...")
        excerpt_f = self.filter(excerpt)

        logger.info("Computing the thresholds...")
        thresholds = compute_threshold(excerpt_f,
                                       single_threshold=single_threshold,
                                       std_factor=std_factor)

        thresholds = {'weak': thresholds[0], 'strong': thresholds[1]}
        logger.info("Thresholds found: {}.".format(thresholds))
        self._thresholder = Thresholder(mode=self.detect_spikes,
                                        thresholds=thresholds)
        return thresholds

    def filter(self, traces):
        f = Filter(rate=self.sample_rate,
                   low=self.filter_low,
                   high=0.95 * .5 * self.sample_rate,
                   order=self.filter_butter_order,
                   )
        return f(traces).astype(np.float32)

    def extract_components(self, filtered):
        # Transform the filtered data according to the detection mode.
        traces_t = self._thresholder.transform(filtered)

        # Compute the threshold crossings.
        weak = self._thresholder.detect(traces_t, 'weak')
        strong = self._thresholder.detect(traces_t, 'strong')

        # Run the detection.
        join_size = self.connected_component_join_size
        detector = FloodFillDetector(probe_adjacency_list=self.adjacency,
                                     join_size=join_size)
        return detector(weak_crossings=weak,
                        strong_crossings=strong)

    def extract_spikes(self, filtered, components):
        # Transform the filtered data according to the detection mode.
        traces_t = self._thresholder.transform(filtered)

        # Extract all waveforms.
        extractor = WaveformExtractor(extract_before=self.extract_s_before,
                                      extract_after=self.extract_s_after,
                                      weight_power=self.weight_power,
                                      thresholds=self._thresholds,
                                      )

        s, m, w = zip(*(extractor(component, data=filtered, data_t=traces_t)
                        for component in components))
        s = np.array(s, dtype=np.int64)
        m = np.array(m, dtype=np.float32)
        w = np.array(w, dtype=np.float32)
        return s, m, w

    def detect(self, traces):
        assert traces.ndim == 2

        # Only keep the selected channels (given shank, no dead channels, etc.)
        traces = self._select_channels(traces)
        assert traces.shape[1] == self.n_channels

        # Find the thresholds.
        self._thresholds = self.find_thresholds(traces)

        # Apply the filter.
        filtered = self.filter(traces)

        # Extract the spike components.
        components = self.extract_components(filtered)

        # Extract the spikes, masks, waveforms.
        spike_samples, masks, waveforms = self.extract_spikes(filtered,
                                                              components)
        return spike_samples, masks
