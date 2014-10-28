# -*- coding: utf-8 -*-
"""Mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Artificial data
#------------------------------------------------------------------------------

def artificial_waveforms(nspikes=None, nsamples=None, nchannels=None):
    # Check arguments.
    assert isinstance(nspikes, six.integer_types)
    assert isinstance(nsamples, six.integer_types)
    assert isinstance(nchannels, six.integer_types)

    # TODO: more realistic waveforms.
    return .25 * np.random.normal(size=(nspikes, nsamples, nchannels))


def artificial_traces(nsamples, nchannels):
    # Check arguments.
    assert isinstance(nsamples, six.integer_types)
    assert isinstance(nchannels, six.integer_types)

    # TODO: more realistic traces.
    return

# def artificial_features(nspikes, nfeatures, use_masks=True):
    # pass
