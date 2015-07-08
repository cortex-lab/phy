# -*- coding: utf-8 -*-

"""PCA for features."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils._types import _as_array


#------------------------------------------------------------------------------
# PCA
#------------------------------------------------------------------------------

def _compute_pcs(x, n_pcs=None, masks=None):
    """Compute the PCs of waveforms."""

    assert x.ndim == 3
    x = _as_array(x, np.float64)

    n_spikes, n_samples, n_channels = x.shape

    if masks is not None:
        assert isinstance(masks, np.ndarray)
        assert masks.shape == (n_spikes, n_channels)

    # Compute regularization cov matrix.
    cov_reg = np.eye(n_samples)
    if masks is not None:
        unmasked = masks > 0
        # The last dimension is now time. The second dimension is channel.
        x_swapped = np.swapaxes(x, 1, 2)
        # This is the list of all unmasked spikes on all channels.
        # shape: (n_unmasked_spikes, n_samples)
        unmasked_all = x_swapped[unmasked, :]
        # Let's compute the regularization cov matrix of this beast.
        # shape: (n_samples, n_samples)
        cov_reg_ = np.cov(unmasked_all, rowvar=0)
        # Make sure the covariance matrix is valid.
        if cov_reg_.ndim == 2:
            cov_reg = cov_reg
    assert cov_reg.shape == (n_samples, n_samples)

    pcs_list = []
    # Loop over channels
    for channel in range(n_channels):
        x_channel = x[:, :, channel]
        # Compute cov matrix for the channel
        if masks is not None:
            # Unmasked waveforms on that channel
            # shape: (n_unmasked, n_samples)
            x_channel = np.compress(masks[:, channel] > 0,
                                    x_channel, axis=0)
        assert x_channel.ndim == 2
        # Don't compute the cov matrix if there are no unmasked spikes
        # on that channel.
        alpha = 1. / n_spikes
        if x_channel.shape[0] <= 1:
            cov = alpha * cov_reg
        else:
            cov_channel = np.cov(x_channel, rowvar=0)
            assert cov_channel.shape == (n_samples, n_samples)
            cov = alpha * cov_reg + cov_channel
        # Compute the eigenelements
        vals, vecs = np.linalg.eigh(cov)
        pcs = vecs.T.astype(np.float32)[np.argsort(vals)[::-1]]
        # Take the first n_pcs components.
        if n_pcs is not None:
            pcs = pcs[:n_pcs, ...]
        pcs_list.append(pcs[:n_pcs, ...])

    pcs = np.dstack(pcs_list)
    return pcs


def _project_pcs(x, pcs):
    """Project data points onto principal components.

    Parameters
    ----------

    x : array
        The waveforms
    pcs : array
        The PCs returned by `_compute_pcs()`.
    """
    # pcs: (nf, ns, nc)
    # x: (n, ns, nc)
    # out: (n, nc, nf)
    assert pcs.ndim == 3
    assert x.ndim == 3
    n, ns, nc = x.shape
    nf, ns_, nc_ = pcs.shape
    assert ns == ns_
    assert nc == nc_

    x_proj = np.einsum('ijk,...jk->...ki', pcs, x)
    assert x_proj.shape == (n, nc, nf)
    return x_proj


class PCA(object):
    """Apply PCA to waveforms."""
    def __init__(self, n_pcs=None):
        self._n_pcs = n_pcs
        self._pcs = None

    def fit(self, waveforms, masks=None):
        """Compute the PCs of waveforms.

        Parameters
        ----------

        waveforms : ndarray
            Shape: `(n_spikes, n_samples, n_channels)`
        masks : ndarray
            Shape: `(n_spikes, n_channels)`

        """
        self._pcs = _compute_pcs(waveforms, n_pcs=self._n_pcs, masks=masks)
        return self._pcs

    def transform(self, waveforms, pcs=None):
        """Project waveforms on the PCs.

        Parameters
        ----------

        waveforms : ndarray
            Shape: `(n_spikes, n_samples, n_channels)`

        """
        if pcs is None:
            pcs = self._pcs
        # Need to call fit() if the pcs are None here.
        if pcs is not None:
            return _project_pcs(waveforms, pcs)
