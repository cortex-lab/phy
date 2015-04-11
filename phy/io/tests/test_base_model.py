# -*- coding: utf-8 -*-

"""Tests of the BaseModel class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..base_model import BaseModel


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_base_model():
    model = BaseModel()

    assert model.channel_group is None

    model.channel_group = 1
    assert model.channel_group == 1

    assert model.channel_groups == []
    assert model.clusterings == []

    model.clustering = 'original'
    assert model.clustering == 'original'

    with raises(NotImplementedError):
        model.metadata
    with raises(NotImplementedError):
        model.traces
    with raises(NotImplementedError):
        model.spike_samples
    with raises(NotImplementedError):
        model.spike_clusters
    with raises(NotImplementedError):
        model.cluster_metadata
    with raises(NotImplementedError):
        model.features
    with raises(NotImplementedError):
        model.masks
    with raises(NotImplementedError):
        model.waveforms
    with raises(NotImplementedError):
        model.probe
    with raises(NotImplementedError):
        model.save()

    model.close()
