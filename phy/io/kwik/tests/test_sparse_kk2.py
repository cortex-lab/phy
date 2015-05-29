# -*- coding: utf-8 -*-

"""Tests of sparse KK2 routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_array_almost_equal as aee

from ..sparse_kk2 import sparsify_features_masks


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_sparsify_features_masks():
    # Data as arrays
    fet = np.array([[1, 3, 5, 7, 11],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]], dtype=float)

    fmask = np.array([[1, 0.5, 0, 0, 0],
                      [0, 1, 1, 0, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 1]], dtype=float)

    # Normalisation to [0, 1]
    fet_original = fet.copy()
    fet = (fet - np.amin(fet, axis=0)) / \
        (np.amax(fet, axis=0) - np.amin(fet, axis=0))

    nanmasked_fet = fet.copy()
    nanmasked_fet[fmask > 0] = np.nan

    # Correct computation of the corrected data and correction term
    x = fet
    w = fmask
    nu = np.nanmean(nanmasked_fet, axis=0)[np.newaxis, :]
    sigma2 = np.nanvar(nanmasked_fet, axis=0)[np.newaxis, :]
    y = w * x + (1 - w) * nu
    z = w * x * x + (1 - w) * (nu * nu + sigma2)
    correction_terms = z - y * y
    features = y

    data = sparsify_features_masks(fet_original, fmask)

    aee(data.noise_mean, np.nanmean(nanmasked_fet, axis=0))
    aee(data.noise_variance, np.nanvar(nanmasked_fet, axis=0))
    assert np.amin(data.features) == 0
    assert np.amax(data.features) == 1
    assert len(data.offsets) == 5

    for i in range(4):
        data_u = data.unmasked[data.offsets[i]:data.offsets[i + 1]]
        true_u, = fmask[i, :].nonzero()
        ae(data_u, true_u)
        data_f = data.features[data.offsets[i]:data.offsets[i + 1]]
        true_f = fet[i, data_u]
        ae(data_f, true_f)
        data_m = data.masks[data.offsets[i]:data.offsets[i + 1]]
        true_m = fmask[i, data_u]
        ae(data_m, true_m)

    # PART 2: Check that converting to SparseData is correct
    # compute unique masks and apply correction terms to data
    data = data.to_sparse_data()

    assert data.num_spikes == 4
    assert data.num_features == 5
    assert data.num_masks == 3

    for i in range(4):
        data_u = data.unmasked[data.unmasked_start[i]:data.unmasked_end[i]]
        true_u, = fmask[i, :].nonzero()
        ae(data_u, true_u)
        data_f = data.features[data.values_start[i]:data.values_end[i]]
        true_f = features[i, data_u]
        aee(data_f, true_f)
        data_c = data.correction_terms[data.values_start[i]:data.values_end[i]]
        true_c = correction_terms[i, data_u]
        aee(data_c, true_c)
        data_m = data.masks[data.values_start[i]:data.values_end[i]]
        true_m = fmask[i, data_u]
        aee(data_m, true_m)
