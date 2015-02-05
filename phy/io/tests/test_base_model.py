# -*- coding: utf-8 -*-

"""Tests of the BaseModel class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises
import numpy as np

from ..base_model import BaseModel


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_base_model():
    model = BaseModel()

    assert model.channel_group is None
    assert model.recording is None

    model.channel_group = 1
    assert model.channel_group == 1

    assert model.channel_groups == []
    assert model.recordings == []
    assert model.clusterings == []

    model.recording = 2
    assert model.recording == 2

    model.clustering = 'original'
    assert model.clustering == 'original'

    with raises(NotImplementedError):
        model.metadata
    with raises(NotImplementedError):
        model.traces
    with raises(NotImplementedError):
        model.spike_times
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
