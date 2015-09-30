# -*- coding: utf-8 -*-

"""Spike detection."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from traitlets.config.configurable import Configurable
from traitlets import Int, Float, Unicode, Bool

from phy.electrode.mea import MEA, _adjacency_subset, _remap_adjacency
from phy.utils.array import get_excerpts
from .detect import FloodFillDetector, Thresholder, compute_threshold
from .filter import Filter
from .waveform import WaveformExtractor

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# SpikeDetector
#------------------------------------------------------------------------------

def _concat_spikes(s, m, w, chunks=None):
    # TODO: overlap
    def add_offset(x, block_id=None):
        i = block_id[0]
        return x + sum(chunks[0][:i])

    s = s.map_blocks(add_offset)
    return s, m, w


class SpikeDetector(Configurable):
    do_filter = Bool(True)
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
        self.set_context(ctx)

    def set_context(self, ctx):
        self.ctx = ctx
        if not ctx or not hasattr(ctx, 'cache'):
            return
        # self.find_thresholds = ctx.cache(self.find_thresholds)
        # self.filter = ctx.cache(self.filter)
        # self.extract_spikes = ctx.cache(self.extract_spikes)
        # self.detect = ctx.cache(self.detect)

    def set_metadata(self, probe, channel_mapping=None,
                     sample_rate=None):
        assert isinstance(probe, MEA)
        self.probe = probe

        assert sample_rate > 0
        self.sample_rate = sample_rate

        # Channel mapping.
        if channel_mapping is None:
            channel_mapping = {c: c for c in probe.channels}
        # channel mappings is {trace_col: channel_id}.
        # Trace columns and channel ids to keep.
        self.trace_cols = sorted(channel_mapping.keys())
        self.channel_ids = sorted(channel_mapping.values())
        # The key is the col in traces, the val is the channel id.
        adj = self.probe.adjacency  # Numbers are all channel ids.
        # First, we subset the adjacency list with the kept channel ids.
        adj = _adjacency_subset(adj, self.channel_ids)
        # Then, we remap to convert from channel ids to trace columns.
        # We need to inverse the mapping.
        channel_mapping_inv = {v: c for (c, v) in channel_mapping.items()}
        # Now, the adjacency list contains trace column numbers.
        adj = _remap_adjacency(adj, channel_mapping_inv)
        assert set(adj) <= set(self.trace_cols)
        # Finally, we need to remap with relative column indices.
        rel_mapping = {c: i for (i, c) in enumerate(self.trace_cols)}
        adj = _remap_adjacency(adj, rel_mapping)
        self._adjacency = adj

        # Array of channel idx to consider.
        self.n_channels = len(self.channel_ids)
        self.n_samples_waveforms = self.extract_s_before + self.extract_s_after

    def subset_traces(self, traces):
        return traces[:, self.trace_cols]

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
        # logger.info("Thresholds found: {}.".format(thresholds))
        return thresholds

    def filter(self, traces):
        if not self.do_filter:  # pragma: no cover
            return traces
        f = Filter(rate=self.sample_rate,
                   low=self.filter_low,
                   high=0.95 * .5 * self.sample_rate,
                   order=self.filter_butter_order,
                   )
        logger.info("Filtering %d samples...", traces.shape[0])
        return f(traces).astype(np.float32)

    def extract_spikes(self, traces_subset, thresholds=None):
        thresholds = thresholds or self._thresholds
        assert thresholds is not None
        self._thresholder = Thresholder(mode=self.detect_spikes,
                                        thresholds=thresholds)

        # Filter the traces.
        traces_f = self.filter(traces_subset)

        # Transform the filtered data according to the detection mode.
        traces_t = self._thresholder.transform(traces_f)

        # Compute the threshold crossings.
        weak = self._thresholder.detect(traces_t, 'weak')
        strong = self._thresholder.detect(traces_t, 'strong')

        # Run the detection.
        logger.info("Detecting connected components...")
        join_size = self.connected_component_join_size
        detector = FloodFillDetector(probe_adjacency_list=self._adjacency,
                                     join_size=join_size)
        components = detector(weak_crossings=weak,
                              strong_crossings=strong)

        # Extract all waveforms.
        extractor = WaveformExtractor(extract_before=self.extract_s_before,
                                      extract_after=self.extract_s_after,
                                      weight_power=self.weight_power,
                                      thresholds=thresholds,
                                      )

        logger.info("Extracting %d spikes...", len(components))
        s, m, w = zip(*(extractor(component, data=traces_f, data_t=traces_t)
                        for component in components))
        s = np.array(s, dtype=np.int64)
        m = np.array(m, dtype=np.float32)
        w = np.array(w, dtype=np.float32)
        return s, m, w

    def detect(self, traces, thresholds=None):

        # Only keep the selected channels (given shank, no dead channels, etc.)
        traces = self.subset_traces(traces)
        assert traces.ndim == 2
        assert traces.shape[1] == self.n_channels

        # Find the thresholds.
        if thresholds is None:
            thresholds = self.find_thresholds(traces)

        # Extract the spikes, masks, waveforms.
        if not self.ctx:
            return self.extract_spikes(traces, thresholds=thresholds)
        else:
            names = ('spike_samples', 'masks', 'waveforms')
            self._thresholds = thresholds
            s, m, w = self.ctx.map_dask_array(self.extract_spikes,
                                              traces, name=names)
            return _concat_spikes(s, m, w, chunks=traces.chunks)
