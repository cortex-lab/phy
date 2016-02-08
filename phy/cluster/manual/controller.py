# -*- coding: utf-8 -*-

"""Controller: model -> views."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np

from phy.cluster.manual.gui_component import ManualClustering
from phy.cluster.manual.views import (WaveformView,
                                      TraceView,
                                      FeatureView,
                                      CorrelogramView,
                                      select_traces,
                                      extract_spikes,
                                      )
from phy.io.array import _get_data_lim, concat_per_cluster
from phy.io import Context, Selector
from phy.stats.clusters import (mean,
                                get_waveform_amplitude,
                                )
from phy.utils import Bunch

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

class Controller(object):
    """Take data out of the model and feeds it to views."""
    # responsible for the cache
    def __init__(self):
        self._init_data()
        self._init_selector()
        self._init_context()

        self.n_spikes = len(self.spike_times)

    # Internal methods
    # -------------------------------------------------------------------------

    def _init_data(self):  # pragma: no cover
        self.cache_dir = None
        # Child classes must set these variables.
        self.spike_times = None  # (n_spikes,) array

        # TODO: make sure these structures are updated during a session
        self.spike_clusters = None  # (n_spikes,) array
        self.cluster_groups = None  # dict {cluster_id: None/'noise'/'mua'}
        self.cluster_ids = None

        self.channel_positions = None  # (n_channels, 2) array
        self.n_samples_waveforms = None  # int > 0
        self.n_channels = None  # int > 0
        self.n_features_per_channel = None  # int > 0
        self.sample_rate = None  # float
        self.duration = None  # float

        self.all_masks = None  # (n_spikes, n_channels)
        self.all_waveforms = None  # (n_spikes, n_samples, n_channels)
        self.all_features = None  # (n_spikes, n_channels, n_features)
        self.all_traces = None  # (n_samples_traces, n_channels)

    def _init_selector(self):
        self.selector = Selector(self.spikes_per_cluster)

    def _init_context(self):
        assert self.cache_dir
        self.context = Context(self.cache_dir)
        ctx = self.context

        self.get_masks = concat_per_cluster(ctx.cache(self.get_masks))
        self.get_features = concat_per_cluster(ctx.cache(self.get_features))
        self.get_waveforms = concat_per_cluster(ctx.cache(self.get_waveforms))

        self.get_background_features = ctx.cache(self.get_background_features)

        self.get_mean_masks = ctx.memcache(self.get_mean_masks)
        self.get_mean_features = ctx.memcache(self.get_mean_features)
        self.get_mean_waveforms = ctx.memcache(self.get_mean_waveforms)

        self.get_waveform_lims = ctx.cache(self.get_waveform_lims)
        self.get_feature_lim = ctx.cache(self.get_feature_lim)

        self.get_waveform_amplitude = ctx.memcache(ctx.cache(
            self.get_waveforms_amplitude))
        self.get_best_channel_position = ctx.memcache(
            self.get_best_channel_position)
        self.get_close_clusters = ctx.memcache(ctx.cache(
            self.get_close_clusters))

        self.spikes_per_cluster = ctx.memcache(self.spikes_per_cluster)

    def _select_spikes(self, cluster_id, n_max=None):
        assert isinstance(cluster_id, int)
        assert cluster_id >= 0
        return self.selector.select_spikes([cluster_id], n_max)

    def _select_data(self, cluster_id, arr, n_max=None):
        spike_ids = self._select_spikes(cluster_id, n_max)
        b = Bunch()
        b.data = arr[spike_ids]
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    def _data_lim(self, arr, n_max):
        return _get_data_lim(arr, n_spikes=n_max)

    # Masks
    # -------------------------------------------------------------------------

    # Is cached in _init_context()
    def get_masks(self, cluster_id):
        return self._select_data(cluster_id,
                                 self.all_masks,
                                 100,  # TODO
                                 )

    def get_mean_masks(self, cluster_id):
        return mean(self.get_masks(cluster_id).data)

    # Waveforms
    # -------------------------------------------------------------------------

    # Is cached in _init_context()
    def get_waveforms(self, cluster_id):
        return [self._select_data(cluster_id,
                                  self.all_waveforms,
                                  100,  # TODO
                                  )]

    def get_mean_waveforms(self, cluster_id):
        return mean(self.get_waveforms(cluster_id)[0].data)

    def get_waveform_lims(self):
        n_spikes = 100  # TODO
        arr = self.all_waveforms
        n = arr.shape[0]
        k = max(1, n // n_spikes)
        # Extract waveforms.
        arr = arr[::k]
        # Take the corresponding masks.
        masks = self.all_masks[::k].copy()
        arr = arr * masks[:, np.newaxis, :]
        # NOTE: on some datasets, there are a few outliers that screw up
        # the normalization. These parameters should be customizable.
        m = np.percentile(arr, .05)
        M = np.percentile(arr, 99.95)
        return m, M

    def get_waveforms_amplitude(self, cluster_id):
        mm = self.get_mean_masks(cluster_id)
        mw = self.get_mean_waveforms(cluster_id)
        assert mw.ndim == 2
        return get_waveform_amplitude(mm, mw)

    # Features
    # -------------------------------------------------------------------------

    # Is cached in _init_context()
    def get_features(self, cluster_id):
        return self._select_data(cluster_id,
                                 self.all_features,
                                 1000,  # TODO
                                 )

    def get_background_features(self):
        k = max(1, int(self.n_spikes // 1000))
        spike_ids = slice(None, None, k)
        b = Bunch()
        b.data = self.all_features[spike_ids]
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    def get_mean_features(self, cluster_id):
        return mean(self.get_features(cluster_id).data)

    def get_feature_lim(self):
        return self._data_lim(self.all_features, 100)  # TODO

    # Traces
    # -------------------------------------------------------------------------

    def get_traces(self, interval):
        tr = select_traces(self.all_traces, interval,
                           sample_rate=self.sample_rate,
                           )
        return [Bunch(traces=tr)]

    def get_spikes_traces(self, interval, traces):
        # NOTE: we extract the spikes from the first traces array.
        traces = traces[0].traces
        b = extract_spikes(traces, interval,
                           sample_rate=self.sample_rate,
                           spike_times=self.spike_times,
                           spike_clusters=self.spike_clusters,
                           all_masks=self.all_masks,
                           n_samples_waveforms=self.n_samples_waveforms,
                           )
        return b

    # Cluster statistics
    # -------------------------------------------------------------------------

    def get_best_channel(self, cluster_id):
        wa = self.get_waveforms_amplitude(cluster_id)
        return int(wa.argmax())

    def get_best_channels(self, cluster_ids):
        channels = [self.get_best_channel(cluster_id)
                    for cluster_id in cluster_ids]
        return list(set(channels))

    def get_channels_by_amplitude(self, cluster_ids):
        wa = self.get_waveforms_amplitude(cluster_ids[0])
        return np.argsort(wa)[::-1].tolist()

    def get_best_channel_position(self, cluster_id):
        cha = self.get_best_channel(cluster_id)
        return tuple(self.channel_positions[cha])

    def get_probe_depth(self, cluster_id):
        return self.get_best_channel_position(cluster_id)[1]

    def get_close_clusters(self, cluster_id):
        assert isinstance(cluster_id, int)
        # Position of the cluster's best channel.
        pos0 = self.get_best_channel_position(cluster_id)
        n = len(pos0)
        assert n in (2, 3)
        # Positions of all clusters' best channels.
        clusters = self.cluster_ids
        pos = np.vstack([self.get_best_channel_position(int(clu))
                         for clu in clusters])
        assert pos.shape == (len(clusters), n)
        # Distance of all clusters to the current cluster.
        dist = (pos - pos0) ** 2
        assert dist.shape == (len(clusters), n)
        dist = np.sum(dist, axis=1) ** .5
        assert dist.shape == (len(clusters),)
        # Closest clusters.
        ind = np.argsort(dist)
        ind = ind[:100]  # TODO
        return [(int(clusters[i]), float(dist[i])) for i in ind]

    def spikes_per_cluster(self, cluster_id):
        return np.nonzero(self.spike_clusters == cluster_id)[0]

    # View methods
    # -------------------------------------------------------------------------

    def add_waveform_view(self, gui):
        v = WaveformView(waveforms=self.get_waveforms,
                         channel_positions=self.channel_positions,
                         waveform_lims=self.get_waveform_lims(),
                         best_channels=self.get_best_channels,
                         )
        v.attach(gui)
        return v

    def add_trace_view(self, gui):
        v = TraceView(traces=self.get_traces,
                      spikes=self.get_spikes_traces,
                      sample_rate=self.sample_rate,
                      duration=self.duration,
                      n_channels=self.n_channels,
                      )
        v.attach(gui)
        return v

    def add_feature_view(self, gui):
        v = FeatureView(features=self.get_features,
                        background_features=self.get_background_features(),
                        spike_times=self.spike_times,
                        n_channels=self.n_channels,
                        n_features_per_channel=self.n_features_per_channel,
                        feature_lim=self.get_feature_lim(),
                        best_channels=self.get_channels_by_amplitude,
                        )
        v.attach(gui)
        return v

    def add_correlogram_view(self, gui):
        v = CorrelogramView(spike_times=self.spike_times,
                            spike_clusters=self.spike_clusters,
                            sample_rate=self.sample_rate,
                            )
        v.attach(gui)
        return v

    # GUI methods
    # -------------------------------------------------------------------------

    def set_manual_clustering(self, gui):
        mc = ManualClustering(self.spike_clusters,
                              self.spikes_per_cluster,
                              similarity=self.get_close_clusters,
                              cluster_groups=self.cluster_groups,
                              )
        self.manual_clustering = mc
        mc.add_column(self.get_probe_depth)
        mc.attach(gui)
