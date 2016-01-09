# -*- coding: utf-8 -*-

"""Cluster store."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import wraps
import logging
from operator import itemgetter

import numpy as np

from .array import _accumulate
from phy.utils import Bunch, _as_scalar, _as_scalars

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
        return Bunch(_accumulate([f(c) for c in cluster_ids]))
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

    def get(self, name):
        return self._stats.get(name, None)

    def attach(self, gui):
        gui.register(self, name='cluster_store')
