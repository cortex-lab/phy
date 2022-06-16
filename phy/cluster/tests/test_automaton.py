# -*- coding: utf-8 -*-

"""Test automaton."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..automaton import State, Transition, ClusterInfo, Automaton
from phylib.utils import connect, Bunch, emit


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

N = 30


@fixture
def data():
    _data = [
        {"id": i,
         "n_spikes": N * 10 - 10 * i,
         "group": {2: 'noise', 3: 'noise', 5: 'mua', 8: 'good'}.get(i, None),
         "is_masked": i in (2, 3, 5),
         } for i in range(N)]
    return _data


def first():
    return 0


def last():
    return N - 1


def similar(clusters):
    assert len(clusters) > 0
    return clusters[0] + 1


def new_cluster_id():
    return N


@fixture
def cluster_info():
    return ClusterInfo(
        first=first,
        last=last,
        similar=similar,
        new_cluster_id=new_cluster_id,
    )


#------------------------------------------------------------------------------
# Test automaton
#------------------------------------------------------------------------------

def test_automaton_state_1():
    s = State(clusters=[1, 2])
    assert s.similar == []


def test_automaton_transition_1():
    s = Transition(name='transition', before=None, after=None)


def test_automaton_1(cluster_info):
    a = Automaton(cluster_info)
