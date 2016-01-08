# -*- coding: utf-8 -*-

"""Manual clustering GUI component."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import wraps
import logging
from operator import itemgetter

import numpy as np

from phy.io.array import Selector
from phy.stats.clusters import (mean,
                                get_max_waveform_amplitude,
                                get_mean_masked_features_distance,
                                get_unmasked_channels,
                                get_sorted_main_channels,
                                )
from phy.utils import IPlugin, _as_scalar, _as_scalars

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _get_data_lim(arr, n_spikes=None, percentile=None):
    n = arr.shape[0]
    k = max(1, n // n_spikes) if n_spikes else 1
    arr = np.abs(arr[::k])
    n = arr.shape[0]
    arr = arr.reshape((n, -1))
    return arr.max()


def get_closest_clusters(cluster_id, cluster_ids, sim_func, max_n=None):
    """Return a list of pairs `(cluster, similarity)` sorted by decreasing
    similarity to a given cluster."""
    l = [(_as_scalar(candidate), _as_scalar(sim_func(cluster_id, candidate)))
         for candidate in _as_scalars(cluster_ids)]
    l = sorted(l, key=itemgetter(1), reverse=True)
    return l[:max_n]


def _log(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        logger.log(5, "Compute %s(%s).", f.__name__, str(args))
        return f(*args, **kwargs)
    return wrapped


def _concat(f):
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


# -----------------------------------------------------------------------------
# Cluster statistics
# -----------------------------------------------------------------------------

class ClusterStore(object):
    def __init__(self, context=None):
        self.context = context
        self._stats = {}

    def add(self, f=None, name=None, cache='disk', concat=None):
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
            return lambda _: self.add(_, name=name, cache=cache, concat=concat)
        name = name or f.__name__
        if cache and self.context:
            f = _log(f)
            f = self.context.cache(f, memcache=(cache == 'memory'))
        assert f
        if concat:
            f = _concat(f)
        self._stats[name] = f
        setattr(self, name, f)
        return f

    def attach(self, gui):
        gui.register(self, name='cluster_store')


def create_cluster_store(model, selector=None, context=None):
    cs = ClusterStore(context=context)

    # TODO: make this configurable.
    max_n_spikes_per_cluster = {
        'masks': 1000,
        'features': 1000,
        'background_features_masks': 1000,
        'waveforms': 100,
        'waveform_lim': 1000,  # used to compute the waveform bounds
        'feature_lim': 1000,  # used to compute the waveform bounds
        'mean_traces': 10000,
    }
    max_n_similar_clusters = 20

    def select(cluster_id, n=None):
        assert isinstance(cluster_id, int)
        assert cluster_id >= 0
        return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)

    # Model data.
    # -------------------------------------------------------------------------

    @cs.add(concat=True)
    def masks(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['masks'])
        if model.masks is None:
            masks = np.ones((len(spike_ids), model.n_channels))
        else:
            masks = np.atleast_2d(model.masks[spike_ids])
        assert masks.ndim == 2
        return spike_ids, masks

    @cs.add(concat=True)
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

    @cs.add(concat=True)
    def features(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['features'])
        if model.features is None:
            features = np.zeros((len(spike_ids),
                                 model.n_channels,
                                 model.n_features_per_channel,
                                 ))
        else:
            features = np.atleast_2d(model.features[spike_ids])
        assert features.ndim == 3
        return spike_ids, features

    @cs.add
    def feature_lim():
        """Return the max of a subset of the feature amplitudes."""
        return _get_data_lim(model.features,
                             max_n_spikes_per_cluster['feature_lim'])

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

    @cs.add(concat=True)
    def waveforms(cluster_id):
        spike_ids = select(cluster_id,
                           max_n_spikes_per_cluster['waveforms'])
        waveforms = np.atleast_2d(model.waveforms[spike_ids])
        assert waveforms.ndim == 3
        return spike_ids, waveforms

    @cs.add
    def waveform_lim():
        """Return the max of a subset of the waveform amplitudes."""
        return _get_data_lim(model.waveforms,
                             max_n_spikes_per_cluster['waveform_lim'])

    @cs.add(concat=True)
    def waveforms_masks(cluster_id):
        spike_ids = select(cluster_id,
                           max_n_spikes_per_cluster['waveforms'])
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
        return np.asscalar(get_max_waveform_amplitude(mm, mw))

    @cs.add(cache=None)
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

    @cs.add(cache='memory')
    def most_similar_clusters(cluster_id):
        assert isinstance(cluster_id, int)
        return get_closest_clusters(cluster_id, model.cluster_ids,
                                    cs.mean_masked_features_score,
                                    max_n_similar_clusters)

    # Traces.
    # -------------------------------------------------------------------------

    @cs.add
    def mean_traces():
        n = max_n_spikes_per_cluster['mean_traces']
        mt = model.traces[:n, :].mean(axis=0)
        return mt.astype(model.traces.dtype)

    return cs


class ClusterStorePlugin(IPlugin):
    def attach_to_gui(self, gui, model=None, state=None):
        ctx = gui.request('context')

        # NOTE: we get the spikes_per_cluster from the Clustering instance.
        # We need to access it from a function to avoid circular dependencies
        # between the cluster store and manual clustering plugins.
        def spikes_per_cluster(cluster_id):
            mc = gui.request('manual_clustering')
            return mc.clustering.spikes_per_cluster[cluster_id]

        assert ctx
        selector = Selector(spike_clusters=model.spike_clusters,
                            spikes_per_cluster=spikes_per_cluster,
                            )
        cs = create_cluster_store(model, selector=selector, context=ctx)
        cs.attach(gui)
