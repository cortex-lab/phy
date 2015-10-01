# -*- coding: utf-8 -*-

"""PCA tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...io.mock import artificial_waveforms, artificial_masks
from ..pca import PCA, _compute_pcs, _project_pcs


#------------------------------------------------------------------------------
# Test PCA
#------------------------------------------------------------------------------

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


def test_project_pcs():
    n, ns, nc = 1000, 50, 100
    nf = 3
    arr = np.random.randn(n, ns, nc)
    pcs = np.random.randn(nf, ns, nc)

    y1 = _project_pcs(arr, pcs)
    assert y1.shape == (n, nc, nf)


class PCATest(object):
    def setup(self):
        self.n_spikes = 100
        self.n_samples = 40
        self.n_channels = 12
        self.waveforms = artificial_waveforms(self.n_spikes,
                                              self.n_samples,
                                              self.n_channels)
        self.masks = artificial_masks(self.n_spikes, self.n_channels)

    def test_serial(self):
        pca = PCA()
        pcs = pca.fit(self.waveforms, self.masks)
        assert pcs.shape == (3, self.n_samples, self.n_channels)
        fet = pca.transform(self.waveforms)
        assert fet.shape == (self.n_spikes, self.n_channels, 3)
