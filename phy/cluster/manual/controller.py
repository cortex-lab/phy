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
from phy.gui import GUI
from phy.io.array import _get_data_lim, concat_per_cluster, get_excerpts
from phy.io import Context, Selector
from phy.plot.transform import _normalize
from phy.stats.clusters import (mean,
                                get_waveform_amplitude,
                                )
from phy.stats.ccg import correlograms
from phy.utils import Bunch, load_master_config, get_plugin, EventEmitter

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

class Controller(EventEmitter):
    """Take data out of the model and feeds it to views.

    Events
    ------

    init()
    create_gui(gui)
    add_view(gui, view)

    """

    gui_name = ''

    n_spikes_waveforms = 100
    n_spikes_waveforms_lim = 100
    n_spikes_masks = 100
    n_spikes_features = 5000
    n_spikes_background_features = 5000
    n_spikes_features_lim = 100
    n_spikes_close_clusters = 100
    n_excerpts_correlograms = 100
    excerpt_size_correlograms = 1000

    # responsible for the cache
    def __init__(self, plugins=None, config_dir=None):
        super(Controller, self).__init__()
        self.config_dir = config_dir
        self._init_data()
        self._init_selector()
        self._init_context()
        self._set_manual_clustering()

        self.n_spikes = len(self.spike_times)

        # Attach the plugins.
        plugins = plugins or []
        config = load_master_config(config_dir=config_dir)
        c = config.get(self.gui_name or self.__class__.__name__)
        default_plugins = c.plugins if c else []
        if len(default_plugins):
            plugins = default_plugins + plugins
        for plugin in plugins:
            try:
                p = get_plugin(plugin)()
            except ValueError:  # pragma: no cover
                logger.warn("The plugin %s couldn't be found.", plugin)
                continue
            try:
                p.attach_to_controller(self)
            except Exception as e:  # pragma: no cover
                logger.warn("An error occurred when attaching plugin %s: %s.",
                            plugin, e)

        self.emit('init')

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
        self.channel_order = None
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
        self.get_waveforms_amplitude = ctx.memcache(
            self.get_waveforms_amplitude)

        self.get_waveform_lims = ctx.memcache(self.get_waveform_lims)
        self.get_feature_lim = ctx.memcache(self.get_feature_lim)
        self.get_trace_lim = ctx.memcache(self.get_trace_lim)

        self.get_close_clusters = ctx.memcache(
            self.get_close_clusters)
        self.get_probe_depth = ctx.memcache(
            self.get_probe_depth)
        self.get_correlograms = ctx.memcache(self.get_correlograms)

        self.spikes_per_cluster = ctx.memcache(self.spikes_per_cluster)

    def _set_manual_clustering(self):
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id'). \
            get('new_cluster_id', None)
        mc = ManualClustering(self.spike_clusters,
                              self.spikes_per_cluster,
                              best_channel=self.get_best_channel,
                              similarity=self.similarity,
                              cluster_groups=self.cluster_groups,
                              new_cluster_id=new_cluster_id,
                              )

        # Save the new cluster id on disk.
        @mc.clustering.connect
        def on_cluster(up):

            # HACK: Update data structures.
            self.cluster_ids = mc.clustering.cluster_ids
            self.spike_clusters = mc.clustering.spike_clusters

            new_cluster_id = mc.clustering.new_cluster_id()
            logger.debug("Save the new cluster id: %d", new_cluster_id)
            self.context.save('new_cluster_id',
                              dict(new_cluster_id=new_cluster_id))

        self.manual_clustering = mc
        mc.add_column(self.get_probe_depth, name='depth')

    def _select_spikes(self, cluster_id, n_max=None, batch_size=None):
        assert isinstance(cluster_id, int)
        assert cluster_id >= 0
        return self.selector.select_spikes([cluster_id], n_max,
                                           batch_size=batch_size)

    def _select_data(self, cluster_id, arr, n_max=None, batch_size=None):
        spike_ids = self._select_spikes(cluster_id, n_max,
                                        batch_size=batch_size)
        b = Bunch()
        b.data = arr[spike_ids]
        b.spike_ids = spike_ids
        b.masks = self.all_masks[spike_ids]
        return b

    def _data_lim(self, arr, n_max=None):
        return _get_data_lim(arr, n_spikes=n_max)

    # Masks
    # -------------------------------------------------------------------------

    # Is cached in _init_context()
    def get_masks(self, cluster_id):
        return self._select_data(cluster_id,
                                 self.all_masks,
                                 self.n_spikes_masks,
                                 )

    def get_mean_masks(self, cluster_id):
        return mean(self.get_masks(cluster_id).data)

    # Waveforms
    # -------------------------------------------------------------------------

    # Is cached in _init_context()
    def get_waveforms(self, cluster_id):
        data = self._select_data(cluster_id,
                                 self.all_waveforms,
                                 self.n_spikes_waveforms,
                                 batch_size=10,
                                 )
        # Sparsify the waveforms.
        w = data.data
        channels = np.nonzero(w.mean(axis=1).mean(axis=0))[0]
        w = w[:, :, channels]
        data.channels = channels
        # Cache the normalized waveforms.
        m, M = self.get_waveform_lims()
        data.data = _normalize(w, m, M)
        data.cluster_id = cluster_id
        return data

    def get_mean_waveforms(self, cluster_id):
        return mean(self.get_waveforms(cluster_id).data)

    def get_waveform_lims(self):
        n_spikes = self.n_spikes_waveforms_lim
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
        a = .05
        m = np.percentile(arr, a)
        M = np.percentile(arr, 100 - a)
        return m, M

    def get_waveforms_amplitude(self, cluster_id):
        mm = self.get_mean_masks(cluster_id)
        mw = self.get_mean_waveforms(cluster_id)
        assert mw.ndim == 2
        return get_waveform_amplitude(mm, mw)

    # Features
    # -------------------------------------------------------------------------

    # Is cached in _init_context()
    def get_features(self, cluster_id, load_all=False):
        data = self._select_data(cluster_id,
                                 self.all_features,
                                 (self.n_spikes_features
                                  if not load_all else None),
                                 )
        m = self.get_feature_lim()
        data.data = _normalize(data.data.copy(), -m, +m)
        return data

    def get_background_features(self):
        k = max(1, int(self.n_spikes // self.n_spikes_background_features))
        spike_ids = slice(None, None, k)
        b = Bunch()
        b.data = self.all_features[spike_ids]
        m = self.get_feature_lim()
        b.data = _normalize(b.data.copy(), -m, +m)
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    def get_mean_features(self, cluster_id):
        return mean(self.get_features(cluster_id).data)

    def get_feature_lim(self):
        return self._data_lim(self.all_features, self.n_spikes_features_lim)

    # Traces
    # -------------------------------------------------------------------------

    def get_traces(self, interval):
        tr = select_traces(self.all_traces, interval,
                           sample_rate=self.sample_rate,
                           )
        return [Bunch(traces=tr * (1. / self.get_trace_lim()))]

    def get_trace_lim(self):
        return self._data_lim(self.all_traces[:10000])

    def get_spikes_traces(self, interval, traces):
        # NOTE: we extract the spikes from the first traces array.
        traces = traces[0].traces
        b = extract_spikes(traces, interval,
                           sample_rate=self.sample_rate,
                           spike_times=self.spike_times,
                           spike_clusters=self.spike_clusters,
                           cluster_groups=self.cluster_groups,
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
        ind = ind[:self.n_spikes_close_clusters]
        return [(int(clusters[i]), float(dist[i])) for i in ind]

    def spikes_per_cluster(self, cluster_id):  # pragma: no cover
        return np.nonzero(self.spike_clusters == cluster_id)[0]

    def get_correlograms(self, cluster_ids, bin_size, window_size):

        # Keep spikes belonging to the selected clusters.
        ind = np.nonzero(np.in1d(self.spike_clusters, cluster_ids))[0]
        ne = self.n_excerpts_correlograms * len(cluster_ids)
        ind = get_excerpts(ind,
                           excerpt_size=self.excerpt_size_correlograms,
                           n_excerpts=ne)

        st = self.spike_times[ind]
        sc = self.spike_clusters[ind]

        # Compute all pairwise correlograms.
        ccg = correlograms(st, sc,
                           cluster_ids=cluster_ids,
                           sample_rate=self.sample_rate,
                           bin_size=bin_size,
                           window_size=window_size,
                           )

        return ccg

    # View methods
    # -------------------------------------------------------------------------

    def _add_view(self, gui, view):
        view.attach(gui)
        self.emit('add_view', gui, view)
        return view

    def add_waveform_view(self, gui):
        v = WaveformView(waveforms=self.get_waveforms,
                         channel_positions=self.channel_positions,
                         channel_order=self.channel_order,
                         best_channels=self.get_best_channels,
                         )
        return self._add_view(gui, v)

    def add_trace_view(self, gui):
        v = TraceView(traces=self.get_traces,
                      spikes=self.get_spikes_traces,
                      sample_rate=self.sample_rate,
                      duration=self.duration,
                      n_channels=self.n_channels,
                      channel_positions=self.channel_positions,
                      channel_order=self.channel_order,
                      )
        return self._add_view(gui, v)

    def add_feature_view(self, gui):
        v = FeatureView(features=self.get_features,
                        background_features=self.get_background_features(),
                        spike_times=self.spike_times,
                        n_channels=self.n_channels,
                        n_features_per_channel=self.n_features_per_channel,
                        feature_lim=self.get_feature_lim(),
                        best_channels=self.get_channels_by_amplitude,
                        )
        return self._add_view(gui, v)

    def add_correlogram_view(self, gui):
        v = CorrelogramView(correlograms=self.get_correlograms,
                            sample_rate=self.sample_rate,
                            )
        return self._add_view(gui, v)

    # GUI methods
    # -------------------------------------------------------------------------

    def similarity(self, cluster_id):
        return self.get_close_clusters(cluster_id)

    def create_gui(self, name=None,
                   subtitle=None,
                   config_dir=None,
                   add_default_views=True,
                   **kwargs):
        """Create a manual clustering GUI."""
        config_dir = config_dir or self.config_dir
        gui = GUI(name=name or self.gui_name,
                  subtitle=subtitle,
                  config_dir=config_dir, **kwargs)
        gui.controller = self

        # Attach the ManualClustering component to the GUI.
        self.manual_clustering.attach(gui)

        # Add views.
        if add_default_views:
            self.add_correlogram_view(gui)
            if self.all_features is not None:
                self.add_feature_view(gui)
            if self.all_waveforms is not None:
                self.add_waveform_view(gui)
            if self.all_traces is not None:
                self.add_trace_view(gui)

        self.emit('create_gui', gui)

        return gui
