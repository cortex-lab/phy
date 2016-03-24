# -*- coding: utf-8 -*-

"""Tests of cluster statistics."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import yield_fixture

from ..clusters import (mean,
                        get_unmasked_channels,
                        get_mean_probe_position,
                        get_sorted_main_channels,
                        get_mean_masked_features_distance,
                        get_waveform_amplitude,
                        )
from phy.electrode.mea import staggered_positions
from phy.io.mock import (artificial_features,
                         artificial_masks,
                         artificial_waveforms,
                         )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def n_channels():
    yield 28


@yield_fixture
def n_spikes():
    yield 50


@yield_fixture
def n_samples():
    yield 40


@yield_fixture
def n_features_per_channel():
    yield 4


@yield_fixture
def features(n_spikes, n_channels, n_features_per_channel):
    yield artificial_features(n_spikes, n_channels, n_features_per_channel)


@yield_fixture
def masks(n_spikes, n_channels):
    yield artificial_masks(n_spikes, n_channels)


@yield_fixture
def waveforms(n_spikes, n_samples, n_channels):
    yield artificial_waveforms(n_spikes, n_samples, n_channels)


@yield_fixture
def site_positions(n_channels):
    yield staggered_positions(n_channels)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_mean(features, n_channels, n_features_per_channel):
    mf = mean(features)
    assert mf.shape == (n_channels, n_features_per_channel)
    ae(mf, features.mean(axis=0))


def test_unmasked_channels(masks, n_channels):
    # Mask many values in the masks array.
    threshold = .05
    masks[:, 1::2] *= threshold
    # Compute the mean masks.
    mean_masks = mean(masks)
    # Find the unmasked channels.
    channels = get_unmasked_channels(mean_masks, threshold)
    # These are 0, 2, 4, etc.
    ae(channels, np.arange(0, n_channels, 2))


def test_mean_probe_position(masks, site_positions):
    masks[:, ::2] *= .05
    mean_masks = mean(masks)
    mean_pos = get_mean_probe_position(mean_masks, site_positions)
    assert mean_pos.shape == (2,)
    assert mean_pos[0] < 0
    assert mean_pos[1] > 0


def test_sorted_main_channels(masks):
    masks *= .05
    masks[:, [5, 7]] *= 20
    mean_masks = mean(masks)
    channels = get_sorted_main_channels(mean_masks,
                                        get_unmasked_channels(mean_masks))
    assert np.all(np.in1d(channels, [5, 7]))


def test_waveform_amplitude(masks, waveforms):
    waveforms *= .1
    masks *= .1

    waveforms[:, 10, :] *= 10
    masks[:, 10] *= 10

    mean_waveforms = mean(waveforms)
    mean_masks = mean(masks)

    amplitude = get_waveform_amplitude(mean_masks, mean_waveforms)
    assert np.all(amplitude >= 0)
    assert amplitude.shape == (mean_waveforms.shape[1],)


def test_mean_masked_features_distance(features,
                                       n_channels,
                                       n_features_per_channel,
                                       ):

    # Shifted feature vectors.
    shift = 10.
    f0 = mean(features)
    f1 = mean(features) + shift

    # Only one channel is unmasked.
    m0 = m1 = np.zeros(n_channels)
    m0[n_channels // 2] = 1

    # Check the distance.
    d_expected = np.sqrt(n_features_per_channel) * shift
    d_computed = get_mean_masked_features_distance(f0, f1, m0, m1,
                                                   n_features_per_channel)
    ac(d_expected, d_computed)
