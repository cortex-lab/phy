# -*- coding: utf-8 -*-

"""PCA tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...io.mock import artificial_waveforms, artificial_masks
from ..pca import PCA, _compute_pcs


#------------------------------------------------------------------------------
# Test PCA
#------------------------------------------------------------------------------

def test_pca():
    n_spikes = 100
    n_samples = 40
    n_channels = 12
    waveforms = artificial_waveforms(n_spikes, n_samples, n_channels)
    masks = artificial_masks(n_spikes, n_channels)

    pca = PCA(n_pcs=3)
    pcs = pca.fit(waveforms, masks)
    assert pcs.shape == (3, n_samples, n_channels)
    fet = pca.transform(waveforms)
    assert fet.shape == (n_spikes, n_channels, 3)


def test_compute_pcs():
    """Test PCA on a 2D array."""
    # Horizontal ellipsoid.
    x = np.random.randn(20000, 2) * np.array([[10., 1.]])
    # Rotate the points by pi/4.
    a = 1. / np.sqrt(2.)
    rot = np.array([[a, -a], [a, a]])
    x = np.dot(x, rot)
    # Compute the PCs.
    pcs = _compute_pcs(x[..., None])
    assert pcs.ndim == 3
    assert (np.abs(pcs) - a).max() < 1e-2


def test_compute_pcs_3d():
    """Test PCA on a 3D array."""
    x1 = np.random.randn(20000, 2) * np.array([[10., 1.]])
    x2 = np.random.randn(20000, 2) * np.array([[1., 10.]])
    x = np.dstack((x1, x2))
    # Compute the PCs.
    pcs = _compute_pcs(x)
    assert pcs.ndim == 3
    assert np.linalg.norm(pcs[0, :, 0] - np.array([-1., 0.])) < 1e-2
    assert np.linalg.norm(pcs[1, :, 0] - np.array([0., -1.])) < 1e-2
    assert np.linalg.norm(pcs[0, :, 1] - np.array([0, 1.])) < 1e-2
    assert np.linalg.norm(pcs[1, :, 1] - np.array([-1., 0.])) < 1e-2
