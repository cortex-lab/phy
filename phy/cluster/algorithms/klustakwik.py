# -*- coding: utf-8 -*-

"""Wrapper to KlustaKwik2 implementation."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import six

from phy.io.array import chunk_bounds
from phy.utils.event import EventEmitter


#------------------------------------------------------------------------------
# Sparse structures
#------------------------------------------------------------------------------

def sparsify_features_masks(features, masks, chunk_size=10000):
    from klustakwik2 import RawSparseData

    assert features.ndim == 2
    assert masks.ndim == 2
    assert features.shape == masks.shape

    n_spikes, num_features = features.shape

    # Stage 1: read min/max of fet values for normalization
    # and count total number of unmasked features.
    vmin = np.ones(num_features) * np.inf
    vmax = np.ones(num_features) * (-np.inf)
    total_unmasked_features = 0
    for _, _, i, j in chunk_bounds(n_spikes, chunk_size):
        f, m = features[i:j], masks[i:j]
        inds = m > 0
        # Replace the masked values by NaN.
        vmin = np.minimum(np.min(f, axis=0), vmin)
        vmax = np.maximum(np.max(f, axis=0), vmax)
        total_unmasked_features += inds.sum()
    # Stage 2: read data line by line, normalising
    vdiff = vmax - vmin
    vdiff[vdiff == 0] = 1
    fetsum = np.zeros(num_features)
    fet2sum = np.zeros(num_features)
    nsum = np.zeros(num_features)
    all_features = np.zeros(total_unmasked_features)
    all_fmasks = np.zeros(total_unmasked_features)
    all_unmasked = np.zeros(total_unmasked_features, dtype=int)
    offsets = np.zeros(n_spikes + 1, dtype=int)
    curoff = 0
    for i in six.moves.range(n_spikes):
        fetvals, fmaskvals = (features[i] - vmin) / vdiff, masks[i]
        inds = (fmaskvals > 0).nonzero()[0]
        masked_inds = (fmaskvals == 0).nonzero()[0]
        all_features[curoff:curoff + len(inds)] = fetvals[inds]
        all_fmasks[curoff:curoff + len(inds)] = fmaskvals[inds]
        all_unmasked[curoff:curoff + len(inds)] = inds
        offsets[i] = curoff
        curoff += len(inds)
        fetsum[masked_inds] += fetvals[masked_inds]
        fet2sum[masked_inds] += fetvals[masked_inds] ** 2
        nsum[masked_inds] += 1
    offsets[-1] = curoff

    nsum[nsum == 0] = 1
    noise_mean = fetsum / nsum
    noise_variance = fet2sum / nsum - noise_mean ** 2

    return RawSparseData(noise_mean,
                         noise_variance,
                         all_features,
                         all_fmasks,
                         all_unmasked,
                         offsets,
                         )


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

class KlustaKwik(EventEmitter):
    """KlustaKwik automatic clustering algorithm."""
    def __init__(self, **kwargs):
        super(KlustaKwik, self).__init__()
        self._kwargs = kwargs
        self.__dict__.update(kwargs)
        # Set the version.
        from klustakwik2 import __version__
        self.version = __version__

    def cluster(self,
                spike_ids=None,
                features=None,
                masks=None,
                ):
        """Run the clustering algorithm on the model, or on any features
        and masks.

        Return the `spike_clusters` assignments.

        Emit the `iter` event at every KlustaKwik iteration.

        """
        # Select some spikes if needed.
        if spike_ids is not None:
            features = features[spike_ids]
            masks = masks[spike_ids]
        # Convert the features and masks to the sparse structure used
        # by KK.
        data = sparsify_features_masks(features, masks)
        data = data.to_sparse_data()
        # Run KK.
        from klustakwik2 import KK
        kk = KK(data, **self._kwargs)

        @kk.register_callback
        def f(_):
            # Skip split iterations.
            if _.name != '':
                return
            self.emit('iter', kk.clusters)

        self.params = kk.all_params
        kk.cluster_mask_starts()
        spike_clusters = kk.clusters
        return spike_clusters


def cluster(features=None, masks=None, algorithm='klustakwik',
            spike_ids=None, **kwargs):
    """Launch an automatic clustering algorithm on the model.

    Parameters
    ----------

    features : ndarray
    masks : ndarray
    algorithm : str
        Only 'klustakwik' is supported currently.
    **kwargs
        Parameters for KK.

    """
    assert algorithm == 'klustakwik'
    kk = KlustaKwik(**kwargs)
    return kk.cluster(features=features, masks=masks, spike_ids=spike_ids)
