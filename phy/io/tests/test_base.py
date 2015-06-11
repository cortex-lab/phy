# -*- coding: utf-8 -*-

"""Tests of the BaseModel class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..base import BaseModel, ClusterMetadata


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_base_cluster_metadata():
    meta = ClusterMetadata()

    @meta.default
    def group(cluster):
        return 3

    @meta.default
    def color(cluster):
        return 0

    assert meta.group(0) is not None
    assert meta.group(2) == 3
    assert meta.group(10) == 3

    meta.set_color(10, 5)
    assert meta.color(10) == 5

    # Alternative __setitem__ syntax.
    meta.set_color([10, 11], 5)
    assert meta.color(10) == 5
    assert meta.color(11) == 5

    meta.set_color([10, 11], 6)
    assert meta.color(10) == 6
    assert meta.color(11) == 6
    assert meta.color([10, 11]) == [6, 6]

    meta.set_color(10, 20)
    assert meta.color(10) == 20


def test_base():
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
