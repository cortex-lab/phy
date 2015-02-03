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
    exp = BaseModel()

    assert exp.channel_group is None
    assert exp.recording is None

    exp.channel_group = 1
    assert exp.channel_group == 1

    exp.recording = 2
    assert exp.recording == 2

    with raises(NotImplementedError):
        exp.metadata
    with raises(NotImplementedError):
        exp.traces
    with raises(NotImplementedError):
        exp.spike_times
    with raises(NotImplementedError):
        exp.spike_clusters
    with raises(NotImplementedError):
        exp.cluster_metadata
    with raises(NotImplementedError):
        exp.features
    with raises(NotImplementedError):
        exp.masks
    with raises(NotImplementedError):
        exp.waveforms
    with raises(NotImplementedError):
        exp.probe
    with raises(NotImplementedError):
        exp.save()
