# -*- coding: utf-8 -*-

"""Cluster statistics."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Cluster statistics
#------------------------------------------------------------------------------

def mean(x):
    return x.mean(axis=0)


def get_unmasked_channels(mean_masks, min_mask=.25):
    return np.nonzero(mean_masks > min_mask)[0]


def get_mean_probe_position(mean_masks, site_positions):
    return (np.sum(site_positions * mean_masks[:, np.newaxis], axis=0) /
            max(1, np.sum(mean_masks)))


def get_sorted_main_channels(mean_masks, unmasked_channels):
    # Weighted mean of the channels, weighted by the mean masks.
    main_channels = np.argsort(mean_masks)[::-1]
    main_channels = np.array([c for c in main_channels
                              if c in unmasked_channels])
    return main_channels


#------------------------------------------------------------------------------
# Wizard measures
#------------------------------------------------------------------------------

def get_max_waveform_amplitude(mean_masks, mean_waveforms):

    assert mean_waveforms.ndim == 2
    n_samples, n_channels = mean_waveforms.shape

    assert mean_masks.ndim == 1
    assert mean_masks.shape == (n_channels,)

    mean_waveforms = mean_masks * mean_waveforms

    # Amplitudes.
    m, M = mean_waveforms.min(axis=1), mean_waveforms.max(axis=1)
    return np.max(M - m)


def get_mean_masked_features_distance(mean_features_0,
                                      mean_features_1,
                                      mean_masks_0,
                                      mean_masks_1,
                                      n_features_per_channel=None,
                                      ):
    """Compute the distance between the mean masked features."""

    assert n_features_per_channel > 0

    mu_0 = mean_features_0.ravel()
    mu_1 = mean_features_1.ravel()

    omeg_0 = mean_masks_0
    omeg_1 = mean_masks_1

    omeg_0 = np.repeat(omeg_0, n_features_per_channel)
    omeg_1 = np.repeat(omeg_1, n_features_per_channel)

    d_0 = mu_0 * omeg_0
    d_1 = mu_1 * omeg_1

    return np.linalg.norm(d_0 - d_1)


#------------------------------------------------------------------------------
# Cluster stats object
#------------------------------------------------------------------------------

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
        self._stats[name] = f
        setattr(self, name, f)
