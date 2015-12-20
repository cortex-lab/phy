# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import wraps
import logging

import numpy as np

from phy.io.array import Selector
from phy.stats.clusters import (mean,
                                get_max_waveform_amplitude,
                                get_mean_masked_features_distance,
                                get_unmasked_channels,
                                get_sorted_main_channels,
                                )
from phy.utils import IPlugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _get_data_bounds(arr, n_spikes=None, percentile=None):
    n = arr.shape[0]
    k = max(1, n // n_spikes) if n_spikes else 1
    arr = np.abs(arr[::k])
    n = arr.shape[0]
    arr = arr.reshape((n, -1))
    arr = arr.max(axis=1)
    m = np.percentile(arr, percentile)
    return m


# -----------------------------------------------------------------------------
# Cluster statistics
# -----------------------------------------------------------------------------

class ClusterStore(object):
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
        gui.register(self, name='cluster_store')


class ClusterStorePlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):
        ctx = gui.request('context')
        selector = Selector(spike_clusters=model.spike_clusters,
                            spikes_per_cluster=model.spikes_per_cluster,
                            )
        cs = create_cluster_store(model, selector=selector, context=ctx)
        cs.attach(gui)


def create_cluster_store(model, selector=None, context=None):
    cs = ClusterStore(context=context)

    # TODO: make this configurable.
    max_n_spikes_per_cluster = {
        'masks': 1000,
        'features': 10000,
        'background_features_masks': 10000,
        'waveforms': 100,
        'waveform_lim': 1000,  # used to compute the waveform bounds
        'feature_lim': 1000,  # used to compute the waveform bounds
        'mean_traces': 10000,
    }

    def select(cluster_id, n=None):
        assert cluster_id >= 0
        return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)

    def concat(f):
        """Take a function accepting a single cluster, and return a function
        accepting multiple clusters."""
        @wraps(f)
        def wrapped(cluster_ids):
            # Single cluster.
            if not hasattr(cluster_ids, '__len__'):
                return f(cluster_ids)
            # Concatenate the result of multiple clusters.
            arrs = zip(*(f(c) for c in cluster_ids))
            return tuple(np.concatenate(_, axis=0) for _ in arrs)
        return wrapped

    # Model data.
    # -------------------------------------------------------------------------

    @cs.add
    @concat
    def masks(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['masks'])
        masks = np.atleast_2d(model.masks[spike_ids])
        assert masks.ndim == 2
        return spike_ids, masks

    @cs.add
    @concat
    def features_masks(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['features'])
        fm = np.atleast_3d(model.features_masks[spike_ids])
        ns = fm.shape[0]
        nc = model.n_channels
        nfpc = model.n_features_per_channel
        assert fm.ndim == 3
        f = fm[..., 0].reshape((ns, nc, nfpc))
        m = fm[:, ::nfpc, 1]
        return spike_ids, f, m

    @cs.add
    @concat
    def features(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['features'])
        features = np.atleast_2d(model.features[spike_ids])
        assert features.ndim == 3
        return spike_ids, features

    @cs.add
    def feature_lim(percentile=95):
        """Return the 95% percentile of all feature amplitudes."""
        # TODO: refactor with waveforms and _get_data_bounds
        k = max(1, model.n_spikes // max_n_spikes_per_cluster['feature_lim'])
        w = np.abs(model.features[::k])
        n = w.shape[0]
        w = w.reshape((n, -1))
        w = w.max(axis=1)
        m = np.percentile(w, percentile)
        return m

    @cs.add
    def background_features_masks():
        n = max_n_spikes_per_cluster['background_features_masks']
        k = max(1, model.n_spikes // n)
        features = model.features[::k]
        masks = model.masks[::k]
        spike_ids = np.arange(0, model.n_spikes, k)
        assert spike_ids.shape == (features.shape[0],)
        assert features.ndim == 3
        assert masks.ndim == 2
        assert masks.shape[0] == features.shape[0]
        return spike_ids, features, masks

    @cs.add
    @concat
    def waveforms(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['waveforms'])
        waveforms = np.atleast_2d(model.waveforms[spike_ids])
        assert waveforms.ndim == 3
        return spike_ids, waveforms

    @cs.add
    def waveform_lim(percentile=95):
        """Return the 95% percentile of all waveform amplitudes."""
        k = max(1, model.n_spikes // max_n_spikes_per_cluster['waveform_lim'])
        w = np.abs(model.waveforms[::k])
        n = w.shape[0]
        w = w.reshape((n, -1))
        w = w.max(axis=1)
        m = np.percentile(w, percentile)
        return m

    @cs.add
    @concat
    def waveforms_masks(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['waveforms'])
        waveforms = np.atleast_2d(model.waveforms[spike_ids])
        assert waveforms.ndim == 3
        masks = np.atleast_2d(model.masks[spike_ids])
        assert masks.ndim == 2
        # Ensure that both arrays have the same number of channels.
        assert masks.shape[1] == waveforms.shape[2]
        return spike_ids, waveforms, masks

    # Mean quantities.
    # -------------------------------------------------------------------------

    @cs.add
    def mean_masks(cluster_id):
        # We access [1] because we return spike_ids, masks.
        return mean(cs.masks(cluster_id)[1])

    @cs.add
    def mean_features(cluster_id):
        return mean(cs.features(cluster_id)[1])

    @cs.add
    def mean_waveforms(cluster_id):
        return mean(cs.waveforms(cluster_id)[1])

    # Statistics.
    # -------------------------------------------------------------------------

    @cs.add(cache='memory')
    def best_channels(cluster_id):
        mm = cs.mean_masks(cluster_id)
        uch = get_unmasked_channels(mm)
        return get_sorted_main_channels(mm, uch)

    @cs.add(cache='memory')
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

    # Traces.
    # -------------------------------------------------------------------------

    @cs.add
    def mean_traces():
        n = max_n_spikes_per_cluster['mean_traces']
        mt = model.traces[:n, model.channel_order].mean(axis=0)
        return mt.astype(model.traces.dtype)

    return cs
