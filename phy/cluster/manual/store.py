# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.stats.clusters import (mean,
                                get_max_waveform_amplitude,
                                get_mean_masked_features_distance,
                                get_unmasked_channels,
                                get_sorted_main_channels,
                                )
from phy.utils import IPlugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Cluster statistics
# -----------------------------------------------------------------------------

class ClusterStats(object):
    def __init__(self, context=None):
        self.context = context
        self._stats = {}

    def add(self, f=None, name=None, cache=None):
        """Add a cluster statistic.

        Parameters
        ----------
        f : function
        name : str
        cache : str
            Can be `None` (no cache), `disk`, or `memory`. In the latter case
            the function will also be cached on disk.

        """
        if f is None:
            return lambda _: self.add(_, name=name, cache=cache)
        name = name or f.__name__
        if cache and self.context:
            f = self.context.cache(f, memcache=(cache == 'memory'))
        assert f
        self._stats[name] = f
        setattr(self, name, f)
        return f

    def attach(self, gui):
        gui.register(self, name='cluster_stats')


class ClusterStatsPlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):
        mc = gui.request('manual_clustering')
        if not mc:
            return
        ctx = gui.request('context')
        cs = create_cluster_stats(model, selector=mc.selector, context=ctx)
        cs.attach(gui)


def create_cluster_stats(model, selector=None, context=None,
                         max_n_spikes_per_cluster=1000):
    cs = ClusterStats(context=context)
    ns = max_n_spikes_per_cluster

    def select(cluster_id, n=None):
        assert cluster_id >= 0
        n = n or ns
        return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)

    @cs.add
    def mean_masks(cluster_id):
        spike_ids = select(cluster_id)
        masks = np.atleast_2d(model.masks[spike_ids])
        assert masks.ndim == 2
        return mean(masks)

    @cs.add
    def mean_features(cluster_id):
        spike_ids = select(cluster_id)
        features = np.atleast_2d(model.features[spike_ids])
        assert features.ndim == 3
        return mean(features)

    @cs.add
    def mean_waveforms(cluster_id):
        spike_ids = select(cluster_id, ns // 10)
        waveforms = np.atleast_2d(model.waveforms[spike_ids])
        assert waveforms.ndim == 3
        mw = mean(waveforms)
        return mw

    @cs.add(cache='memory')
    def best_channels(cluster_id):
        mm = cs.mean_masks(cluster_id)
        uch = get_unmasked_channels(mm)
        return get_sorted_main_channels(mm, uch)

    @cs.add
    def best_channels_multiple(cluster_ids):
        best_channels = []
        for cluster in cluster_ids:
            channels = cs.best_channels(cluster)
            best_channels.extend([ch for ch in channels
                                  if ch not in best_channels])
        return best_channels

    @cs.add(cache='memory')
    def max_waveform_amplitude(cluster_id):
        mm = cs.mean_masks(cluster_id)
        mw = cs.mean_waveforms(cluster_id)
        assert mw.ndim == 2
        logger.debug("Computing the quality of cluster %d.", cluster_id)
        return np.asscalar(get_max_waveform_amplitude(mm, mw))

    @cs.add(cache='memory')
    def mean_masked_features_score(cluster_0, cluster_1):
        mf0 = cs.mean_features(cluster_0)
        mf1 = cs.mean_features(cluster_1)
        mm0 = cs.mean_masks(cluster_0)
        mm1 = cs.mean_masks(cluster_1)
        nfpc = model.n_features_per_channel
        d = get_mean_masked_features_distance(mf0, mf1, mm0, mm1,
                                              n_features_per_channel=nfpc)
        s = 1. / max(1e-10, d)
        return s

    return cs
