# -*- coding: utf-8 -*-

"""Tests of the Experiment class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises
import numpy as np

from ..experiment import BaseExperiment


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_base_experiment():
    exp = BaseExperiment()

    with raises(NotImplementedError):
        exp.metadata()
    with raises(NotImplementedError):
        exp.traces()
    with raises(NotImplementedError):
        exp.spike_times()
    with raises(NotImplementedError):
        exp.spike_clusters()
    with raises(NotImplementedError):
        exp.cluster_metadata()
    with raises(NotImplementedError):
        exp.features()
    with raises(NotImplementedError):
        exp.masks()
    with raises(NotImplementedError):
        exp.waveforms()
    with raises(NotImplementedError):
        exp.probe()
    with raises(NotImplementedError):
        exp.save()
